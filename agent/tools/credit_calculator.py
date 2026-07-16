import logging
from typing import Dict, Any

logger = logging.getLogger("agent.tools.credit_calculator")

def credit_audit_calculator(
    gpa: float,
    local_credits: float,
    community_college_transfer: float = 0.0,
    four_year_transfer: float = 0.0
) -> Dict[str, Any]:
    """
    Computes a detailed student academic credit and graduation audit based on university policy.
    
    University Rules:
      1. Good Academic Standing: Cumulative GPA >= 2.0. If GPA < 2.0, status is Academic Probation.
      2. CC Transfer Cap: Max 60 credits accepted from 2-year/community colleges.
      3. 4-Year Transfer Cap: Max 90 credits accepted from 4-year institutions.
      4. Combined Transfer Cap: Max 90 credits accepted total from all external sources.
      5. Residency requirement: Minimum 30 credits must be taken locally in residence.
      6. Graduation minimum: 120 total accepted credits.
    """
    logger.info(f"Running credit audit: GPA={gpa}, Local={local_credits}, CC={community_college_transfer}, 4y={four_year_transfer}")
    
    # 1. Evaluate accepted transfer credits
    accepted_cc = min(max(0.0, community_college_transfer), 60.0)
    cc_overflow = max(0.0, community_college_transfer - 60.0)
    
    accepted_four_year = min(max(0.0, four_year_transfer), 90.0)
    four_year_overflow = max(0.0, four_year_transfer - 90.0)
    
    raw_combined_transfer = accepted_cc + accepted_four_year
    accepted_transfer = min(raw_combined_transfer, 90.0)
    combined_transfer_overflow = max(0.0, raw_combined_transfer - 90.0)
    
    # 2. Total accepted credits
    total_credits = local_credits + accepted_transfer
    
    # 3. Check Academic Standing
    standing = "Good Academic Standing" if gpa >= 2.0 else "Academic Probation"
    
    # 4. Check Residency Requirement
    residency_satisfied = local_credits >= 30.0
    
    # 5. Check Graduation Eligibility
    credits_needed = max(0.0, 120.0 - total_credits)
    meets_credit_minimum = total_credits >= 120.0
    meets_gpa_minimum = gpa >= 2.0
    
    eligible_to_graduate = meets_credit_minimum and meets_gpa_minimum and residency_satisfied
    
    # Generate audit explanation
    status_summary = (
        f"**Audit Status**: {'ELIGIBLE FOR GRADUATION' if eligible_to_graduate else 'INELIGIBLE FOR GRADUATION'}\n"
        f"- **Academic Standing**: {standing} (GPA: {gpa:.2f})\n"
        f"- **Total Accepted Credits**: {total_credits:.1f} / 120.0 minimum\n"
        f"  - Local Credits: {local_credits:.1f} (Residency Requirement: {local_credits:.1f}/30.0 - {'Satisfied' if residency_satisfied else 'Unsatisfied'})\n"
        f"  - Accepted Transfer Credits: {accepted_transfer:.1f} (Out of {community_college_transfer + four_year_transfer:.1f} attempted)\n"
    )
    
    warnings = []
    if cc_overflow > 0:
        warnings.append(f"Attempted CC transfer was {community_college_transfer:.1f} credits, but capped at 60.0 (lost {cc_overflow:.1f} credits).")
    if four_year_overflow > 0:
        warnings.append(f"Attempted 4-Year transfer was {four_year_transfer:.1f} credits, but capped at 90.0 (lost {four_year_overflow:.1f} credits).")
    if combined_transfer_overflow > 0:
        warnings.append(f"Total transfer credits capped at 90.0 combined limit (lost {combined_transfer_overflow:.1f} credits).")
    if local_credits < 30.0:
        warnings.append(f"Fails to meet the 30.0 local credit residency requirement (currently has {local_credits:.1f}).")
    if gpa < 2.0:
        warnings.append("Academic Standing is on Probation (GPA < 2.0). Cumulative GPA of 2.0 is required to graduate.")
        
    explanation = status_summary
    if warnings:
        explanation += "\n**Policy Alerts / Warnings**:\n" + "\n".join([f"* {w}" for w in warnings])
    else:
        explanation += "\nAll credit transfer and academic standing criteria comply with University policy."
        
    return {
        "eligible_to_graduate": eligible_to_graduate,
        "standing": standing,
        "accepted_transfer_credits": accepted_transfer,
        "total_accepted_credits": total_credits,
        "residency_satisfied": residency_satisfied,
        "credits_needed_to_graduate": credits_needed,
        "explanation": explanation
    }


def execute_credit_calculator_tool(tool_input: Dict[str, Any]) -> str:
    """Wrapper that parses inputs and returns a formatted markdown summary."""
    try:
        # LLMs might return strings, parse safely
        gpa = float(tool_input.get("gpa", 0.0))
        local_credits = float(tool_input.get("local_credits", 0.0))
        cc_transfer = float(tool_input.get("community_college_transfer", 0.0))
        fy_transfer = float(tool_input.get("four_year_transfer", 0.0))
        
        audit = credit_audit_calculator(
            gpa=gpa,
            local_credits=local_credits,
            community_college_transfer=cc_transfer,
            four_year_transfer=fy_transfer
        )
        return audit["explanation"]
    except Exception as e:
        logger.error(f"Error running credit calculator tool: {e}")
        return f"Error executing credit audit calculator tool: {str(e)}"
