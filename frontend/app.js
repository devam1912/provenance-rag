document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chat-messages");
    const chatForm = document.getElementById("chat-form");
    const queryInput = document.getElementById("query-input");
    const serverStatus = document.getElementById("server-status");
    const progressTimeline = document.getElementById("progress-timeline");
    const statusText = document.getElementById("timeline-status-text");
    
    // Citation Drawer elements
    const citationDrawer = document.getElementById("citation-drawer");
    const closeDrawerBtn = document.getElementById("close-drawer-btn");
    const inspectChunkId = document.getElementById("inspect-chunk-id");
    const inspectDocName = document.getElementById("inspect-doc-name");
    const inspectExcerptText = document.getElementById("inspect-excerpt-text");

    // Dynamic Upload elements
    const pdfUploadInput = document.getElementById("pdf-upload-input");
    const uploadDocBtn = document.getElementById("upload-doc-btn");
    const uploadStatus = document.getElementById("upload-status");
    const docListContainer = document.getElementById("doc-list");

    // Filter elements
    const activeFilterContainer = document.getElementById("active-filter-container");
    const activeFilterName = document.getElementById("active-filter-name");
    const clearFilterBtn = document.getElementById("clear-filter-btn");

    // Local state
    let chatHistory = [];
    let currentCitationsMap = {}; // Maps chunk_id -> raw content text
    let timelineTimers = [];
    let selectedDocument = null;

    // Fetch and render knowledge base documents
    async function loadDocuments() {
        try {
            const res = await fetch("/api/documents");
            if (!res.ok) throw new Error("Failed to load documents");
            const docs = await res.json();
            
            if (docs && docs.length > 0) {
                docListContainer.innerHTML = "";
                docs.forEach(doc => {
                    const sizeKb = (doc.size_bytes / 1024).toFixed(1);
                    const docItem = document.createElement("div");
                    docItem.className = "doc-item";
                    
                    let description = `Size: ${sizeKb} KB`;
                    if (doc.name === "academic_standing.txt") {
                        description = "GPA rules, Warning, Suspensions";
                    } else if (doc.name === "graduation_requirements.txt") {
                        description = "Residency, Upper-level credits";
                    } else if (doc.name === "transfer_credits.txt") {
                        description = "C- grades, CC cap limit, GPA policy";
                    }
                    
                    docItem.innerHTML = `
                        <span class="doc-icon">📄</span>
                        <div class="doc-details">
                            <div class="doc-name">${doc.name}</div>
                            <div class="doc-desc">${description}</div>
                        </div>
                    `;
                    
                    if (selectedDocument === doc.name) {
                        docItem.classList.add("active");
                    }
                    
                    docItem.addEventListener("click", () => {
                        toggleDocumentFilter(doc.name, docItem);
                    });
                    
                    docListContainer.appendChild(docItem);
                });
            }
        } catch (err) {
            console.error("Error loading document catalog:", err);
        }
    }

    function toggleDocumentFilter(docName, element) {
        const isActive = element.classList.contains("active");
        
        document.querySelectorAll(".doc-item").forEach(item => {
            item.classList.remove("active");
        });

        if (isActive) {
            selectedDocument = null;
            activeFilterContainer.style.display = "none";
        } else {
            selectedDocument = docName;
            element.classList.add("active");
            activeFilterName.textContent = docName;
            activeFilterContainer.style.display = "flex";
        }
    }

    clearFilterBtn.addEventListener("click", () => {
        selectedDocument = null;
        activeFilterContainer.style.display = "none";
        document.querySelectorAll(".doc-item").forEach(item => {
            item.classList.remove("active");
        });
    });

    // Trigger hidden file selection
    uploadDocBtn.addEventListener("click", () => {
        pdfUploadInput.click();
    });

    // Handle file upload
    pdfUploadInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        uploadStatus.style.display = "block";
        uploadStatus.style.color = "var(--text-secondary)";
        uploadStatus.textContent = "Uploading & extracting text...";
        uploadDocBtn.disabled = true;

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Failed to process document");
            }

            uploadStatus.style.color = "var(--teal)";
            uploadStatus.textContent = "Document indexed successfully!";
            
            setTimeout(() => {
                uploadStatus.style.display = "none";
                pdfUploadInput.value = "";
                loadDocuments();
            }, 1800);

        } catch (err) {
            console.error("File upload error:", err);
            uploadStatus.style.color = "#f87171"; 
            uploadStatus.textContent = `Error: ${err.message || "Failed to process document."}`;
        } finally {
            uploadDocBtn.disabled = false;
        }
    });

    // Check Backend Health
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                serverStatus.textContent = "ONLINE";
                serverStatus.className = "status-val success";
            } else {
                throw new Error("Offline");
            }
        } catch (e) {
            serverStatus.textContent = "OFFLINE";
            serverStatus.className = "status-val";
        }
    }
    
    checkHealth();
    loadDocuments();
    // Poll health check every 15s
    setInterval(checkHealth, 15000);

    // Form submission handler
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        submitQuery(query);
    });

    // Handle Quick Suggestions click
    document.addEventListener("click", (e) => {
        if (e.target.classList.contains("suggest-btn")) {
            const query = e.target.getAttribute("data-query");
            if (query) {
                submitQuery(query);
            }
        }
    });

    // Submit Query to API
    async function submitQuery(query) {
        // Clear input field
        queryInput.value = "";
        queryInput.disabled = true;

        // 1. Add User Message
        appendMessage("user", query);
        
        // 2. Show progress timeline and simulate steps
        showTimeline(query);

        // 3. Create dummy loading skeleton for Agent
        const loadingId = appendLoadingSkeleton();

        try {
            const response = await fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    chat_history: chatHistory,
                    filter_document: selectedDocument
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned code ${response.status}`);
            }

            const data = await response.json();
            
            // Remove skeleton loader
            removeElement(loadingId);
            hideTimeline(true);

            // Save citations in local map
            if (data.citations && data.citations.length > 0) {
                data.citations.forEach(c => {
                    currentCitationsMap[c.chunk_id] = c.text;
                });
            }

            // Append Agent response
            appendMessage("agent", data.answer, {
                route: data.route,
                latency: (data.latency_ms / 1000).toFixed(2),
                citations: data.citations
            });

            // Update Chat History
            chatHistory.push({ role: "user", content: query });
            chatHistory.push({ role: "assistant", content: data.answer });
            if (chatHistory.length > 10) {
                chatHistory.shift(); // Keep window size bounded
                chatHistory.shift();
            }

        } catch (err) {
            console.error(err);
            removeElement(loadingId);
            hideTimeline(false);
            appendMessage("agent", "❌ Sorry, I encountered an error while communicating with the academic policy advisory engine. Please verify the server connection and try again.");
        } finally {
            queryInput.disabled = false;
            queryInput.focus();
        }
    }

    // Append standard messages
    function appendMessage(role, text, meta = null) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${role}-message`;

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "message-bubble";
        
        if (role === "user") {
            bubbleDiv.textContent = text;
        } else {
            // Parse response formatting and mark citations as links
            bubbleDiv.innerHTML = formatMarkdownAndCitations(text);
        }
        
        messageDiv.appendChild(bubbleDiv);

        if (meta) {
            const metaDiv = document.createElement("div");
            metaDiv.className = "message-meta";
            
            const routeSpan = document.createElement("span");
            routeSpan.className = "meta-route-label";
            routeSpan.textContent = `Route: ${meta.route}`;
            
            const latencySpan = document.createElement("span");
            latencySpan.textContent = `Latency: ${meta.latency}s`;
            
            metaDiv.appendChild(routeSpan);
            metaDiv.appendChild(latencySpan);
            messageDiv.appendChild(metaDiv);
        }

        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // Append Loading skeleton
    function appendLoadingSkeleton() {
        const id = "loading-" + Date.now();
        const messageDiv = document.createElement("div");
        messageDiv.className = "message agent-message";
        messageDiv.id = id;

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "message-bubble";
        bubbleDiv.style.width = "280px";
        
        bubbleDiv.innerHTML = `
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
        `;
        
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Regex parsing to convert markdown headers, bullet points, and citations
    function formatMarkdownAndCitations(text) {
        let formatted = text;
        
        // Escape HTML tags to prevent cross site scripting
        formatted = formatted.replace(/</g, "&lt;").replace(/>/g, "&gt;");

        // Convert triple hashes to subheaders
        formatted = formatted.replace(/### (.*)/g, "<h3>$1</h3>");
        
        // Convert markdown bold
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

        // Convert bullet lines
        formatted = formatted.replace(/^\* (.*)/gm, "<li>$1</li>");
        formatted = formatted.replace(/^(<li>.*<\/li>)/gms, "<ul>$1</ul>");

        // Convert citation tags [filename#chunk_idx] to clickable numbered badges (e.g., [1], [2])
        // Example tag: [academic_standing.txt#chunk_3]
        let citationIndex = 1;
        const citationIndices = {};
        formatted = formatted.replace(/\`?\[([a-zA-Z0-9_\s\(\)\.-]+#chunk_[0-9]+)\]\`?/g, (match, chunkId) => {
            const trimmedChunk = chunkId.trim();
            if (!citationIndices[trimmedChunk]) {
                citationIndices[trimmedChunk] = citationIndex++;
            }
            const displayIndex = citationIndices[trimmedChunk];
            return `<a class="citation-link" data-chunk-id="${trimmedChunk}">[${displayIndex}]</a>`;
        });

        // Replace newlines with breaks outside of structured tags
        formatted = formatted.replace(/\n/g, "<br>");
        // Fix duplicate breaks inside lists
        formatted = formatted.replace(/<\/li><br>/g, "</li>");
        formatted = formatted.replace(/<\/ul><br>/g, "</ul>");
        formatted = formatted.replace(/<\/h3><br>/g, "</h3>");

        return formatted;
    }

    // Timeline simulation orchestrations
    function showTimeline(query) {
        // Reset steps classes
        document.querySelectorAll(".step").forEach(step => {
            step.className = "step";
        });
        
        progressTimeline.style.display = "block";
        statusText.textContent = "Agent evaluating query structure...";
        
        // Clear previous timers if any
        timelineTimers.forEach(t => clearTimeout(t));
        timelineTimers = [];

        // Step 1: Router immediately active
        const rStep = document.getElementById("step-router");
        rStep.classList.add("active");

        // Step 2: Retrieve & Rerank active after 0.8s
        timelineTimers.push(setTimeout(() => {
            rStep.classList.remove("active");
            rStep.classList.add("completed");
            
            const retStep = document.getElementById("step-retrieve");
            retStep.classList.add("active");
            statusText.textContent = "Retrieving policy documents and reranking relevance...";
        }, 800));

        // Step 3: Tool Execution (if numeric check keywords exist)
        const qLower = query.toLowerCase();
        const hasNumbers = /\d+/.test(qLower);
        const hasKeywords = qLower.includes("gpa") || qLower.includes("credit") || qLower.includes("transfer");
        const isToolCall = hasNumbers && hasKeywords;

        timelineTimers.push(setTimeout(() => {
            const retStep = document.getElementById("step-retrieve");
            retStep.classList.remove("active");
            retStep.classList.add("completed");
            
            const toolStep = document.getElementById("step-tool");
            if (isToolCall) {
                toolStep.classList.add("active");
                statusText.textContent = "Running graduation credit calculator advising audit...";
            } else {
                toolStep.classList.add("completed");
                const synthStep = document.getElementById("step-synthesize");
                synthStep.classList.add("active");
                statusText.textContent = "Synthesizing answers with citations via Gemini...";
            }
        }, 1800));

        // Step 4: Synthesis & Validation
        timelineTimers.push(setTimeout(() => {
            const toolStep = document.getElementById("step-tool");
            toolStep.classList.remove("active");
            toolStep.classList.add("completed");
            
            const synthStep = document.getElementById("step-synthesize");
            if (synthStep.classList.contains("active")) {
                synthStep.classList.remove("active");
                synthStep.classList.add("completed");
            }
            
            const valStep = document.getElementById("step-validate");
            valStep.classList.add("active");
            statusText.textContent = "Validator executing logical post-generation NLI checks...";
        }, 2800));
    }

    function hideTimeline(success) {
        timelineTimers.forEach(t => clearTimeout(t));
        timelineTimers = [];

        if (success) {
            // Mark all steps as complete
            document.querySelectorAll(".step").forEach(step => {
                step.className = "step completed";
            });
            statusText.textContent = "Query validation passed successfully!";
            
            // Hide timeline panel after 1s
            setTimeout(() => {
                progressTimeline.style.display = "none";
            }, 1000);
        } else {
            progressTimeline.style.display = "none";
        }
    }

    // Toggle Citation Inspect panel open
    document.addEventListener("click", (e) => {
        if (e.target.classList.contains("citation-link")) {
            const chunkId = e.target.getAttribute("data-chunk-id");
            if (chunkId) {
                openCitationDrawer(chunkId);
            }
        }
    });

    function openCitationDrawer(chunkId) {
        inspectChunkId.textContent = chunkId;
        
        // Extract document name
        const docName = chunkId.split("#")[0];
        inspectDocName.textContent = docName;
        
        // Load content excerpt from cache map
        const excerptText = currentCitationsMap[chunkId] || "Raw excerpt content is missing or could not be loaded.";
        inspectExcerptText.textContent = excerptText;

        citationDrawer.classList.add("open");
    }

    closeDrawerBtn.addEventListener("click", () => {
        citationDrawer.classList.remove("open");
    });
});
