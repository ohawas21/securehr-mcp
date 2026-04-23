"""
SecureHR MCP Server — Timecount API Tools with RBAC

This FastMCP server exposes 12 Timecount API tools.
RBAC is enforced here: every tool receives a `caller_username` argument
that the FastAPI layer injects into Claude's system prompt so Claude
always passes it through. The USERS_DB is read from the same
CHAT_USERS environment variable as the main FastAPI app.

Deploy this as a SEPARATE Railway service. Its URL goes into
TIMECOUNT_MCP_URL on the main FastAPI service.
"""

import os
import json
import httpx
from typing import Optional
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TIMECOUNT_API_URL   = os.getenv("TIMECOUNT_API_URL", "https://tutorial.formatgold.de/api")
TIMECOUNT_API_TOKEN = os.getenv("TIMECOUNT_API_TOKEN")
MCP_PORT            = int(os.getenv("MCP_PORT", 8001))

if not TIMECOUNT_API_TOKEN:
    raise RuntimeError("TIMECOUNT_API_TOKEN env var is required.")

# ── Shared HTTP client (created once at module load) ──────────────────────────
_http = httpx.AsyncClient(
    base_url=TIMECOUNT_API_URL,
    headers={"Authorization": f"Bearer {TIMECOUNT_API_TOKEN}"},
    timeout=30.0,
)

# ── User DB (same format as main app) ────────────────────────────────────────
def _parse_users() -> dict:
    raw = os.getenv("CHAT_USERS", "")
    users: dict = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) < 2:
            continue
        username    = parts[0].strip()
        password    = parts[1].strip()
        employee_id = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        role        = parts[3].strip() if len(parts) > 3 and parts[3].strip() else "employee"
        dept_id     = int(parts[4].strip()) if len(parts) > 4 and parts[4].strip() else None
        users[username] = {
            "employee_id": employee_id,
            "role": role,
            "department_id": dept_id,
        }
    return users

USERS_DB: dict = _parse_users()

# ── RBAC helpers ──────────────────────────────────────────────────────────────

def _get_role(username: str) -> str:
    return USERS_DB.get(username, {}).get("role", "employee")

def _get_emp_id(username: str) -> Optional[str]:
    return USERS_DB.get(username, {}).get("employee_id")

def _can_read(username: str, target_id: str) -> bool:
    role = _get_role(username)
    if role in ("system_admin", "hr_admin"):
        return True
    return str(_get_emp_id(username)) == str(target_id)

def _can_write(username: str, target_id: str) -> bool:
    return _can_read(username, target_id)   # same rules for now

def _is_admin(username: str) -> bool:
    return _get_role(username) in ("system_admin", "hr_admin")

def _deny(action: str, username: str, resource: str) -> dict:
    print(f"🔒 RBAC DENIED | user={username} action={action} resource={resource}")
    return {
        "error": "Access denied",
        "message": (
            f"You are not authorised to {action} {resource}. "
            f"Your employee ID is: {_get_emp_id(username)}"
        ),
    }

def _log(username: str, action: str, detail: str = ""):
    print(f"📋 AUDIT | user={username} action={action} {detail}")

# ── FastMCP app ───────────────────────────────────────────────────────────────
mcp = FastMCP("SecureHR-Timecount")

# ══════════════════════════════════════════════════════════════════════════════
# EMPLOYEE TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_employees(caller_username: str, visibility: str = "all") -> dict:
    """
    Get a list of employees from Timecount.
    Admins get all employees; regular employees only see their own record.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        visibility: Filter — 'all', '0' (hidden), or '1' (visible).
    """
    _log(caller_username, "get_employees", f"visibility={visibility}")

    if not _is_admin(caller_username):
        # Non-admins: return only own record
        emp_id = _get_emp_id(caller_username)
        if not emp_id:
            return {"error": "Your account is not linked to an employee record."}
        resp = await _http.get(f"/employees/{emp_id}")
        resp.raise_for_status()
        return {"data": [resp.json()]}

    params = {}
    if visibility != "all":
        params["filter[employee_visibility]"] = visibility
    resp = await _http.get("/employees", params=params)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def search_employees(caller_username: str, query: str) -> dict:
    """
    Search for employees by name or token.
    Admins search all; employees only see matches against their own record.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        query: Name or token to search for.
    """
    _log(caller_username, "search_employees", f"query={query}")

    if not _is_admin(caller_username):
        emp_id = _get_emp_id(caller_username)
        if not emp_id:
            return {"data": []}
        resp = await _http.get(f"/employees/{emp_id}")
        resp.raise_for_status()
        emp = resp.json()
        full_name = f"{emp.get('first_name','')} {emp.get('last_name','')}".lower()
        if query.lower() in full_name or query.lower() in str(emp.get("employee_number", "")):
            return {"data": [emp]}
        return {"data": []}

    resp = await _http.get(f"/employees/search/{query}")
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def get_employee_by_id(caller_username: str, employee_id: str) -> dict:
    """
    Get detailed information about a specific employee by their ID.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        employee_id: The employee's numeric ID.
    """
    if not _can_read(caller_username, employee_id):
        return _deny("read", caller_username, f"employee {employee_id}")

    _log(caller_username, "get_employee_by_id", f"id={employee_id}")
    resp = await _http.get(f"/employees/{employee_id}")
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def get_employee_by_token(caller_username: str, token: str) -> dict:
    """
    Get detailed information about a specific employee by their token.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        token: The employee's token string.
    """
    _log(caller_username, "get_employee_by_token", f"token={token}")
    # Fetch first, then RBAC-check on returned ID
    resp = await _http.get(f"/employees/token/{token}")
    resp.raise_for_status()
    emp = resp.json()
    emp_id = str(emp.get("id", ""))
    if not _can_read(caller_username, emp_id):
        return _deny("read", caller_username, f"employee token={token}")
    return emp


@mcp.tool()
async def update_employee(
    caller_username: str,
    employee_id: str,
    # Personal
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    title: Optional[str] = None,
    gender: Optional[int] = None,
    birth_date: Optional[str] = None,
    birth_place: Optional[str] = None,
    birth_name: Optional[str] = None,
    nationality: Optional[str] = None,
    civil_state_id: Optional[str] = None,
    # Contact
    mobile: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    # Address
    street: Optional[str] = None,
    street_no: Optional[str] = None,
    zipcode: Optional[str] = None,
    place: Optional[str] = None,
    country: Optional[str] = None,
    address_addon: Optional[str] = None,
    # Banking
    account_number: Optional[str] = None,
    bank_id: Optional[str] = None,
    payment_method: Optional[int] = None,
    alternative_account_holder: Optional[str] = None,
    # Employment
    department_id: Optional[int] = None,
    employment_id: Optional[str] = None,
    visibility: Optional[int] = None,
    first_entry_date: Optional[str] = None,
    discharge_date: Optional[str] = None,
    # Tax / Insurance
    social_security_number: Optional[str] = None,
    tax_identification_number: Optional[str] = None,
    # Other
    work_permit: Optional[str] = None,
    own_car: Optional[int] = None,
    datev_id: Optional[str] = None,
) -> dict:
    """
    Update an employee record (partial PATCH). Only include fields to change.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        employee_id: The employee's numeric ID (required).
        (all other args): Fields to update — omit to leave unchanged.
    """
    if not _can_write(caller_username, employee_id):
        return _deny("edit", caller_username, f"employee {employee_id}")

    # Build payload — only non-None values
    payload = {k: v for k, v in {
        "first_name": first_name, "last_name": last_name, "title": title,
        "gender": gender, "birth_date": birth_date, "birth_place": birth_place,
        "birth_name": birth_name, "nationality": nationality,
        "civil_state_id": civil_state_id,
        "mobile": mobile, "phone": phone, "email": email,
        "street": street, "street_no": street_no, "zipcode": zipcode,
        "place": place, "country": country, "address_addon": address_addon,
        "account_number": account_number, "bank_id": bank_id,
        "payment_method": payment_method,
        "alternative_account_holder": alternative_account_holder,
        "department_id": department_id, "employment_id": employment_id,
        "visibility": visibility, "first_entry_date": first_entry_date,
        "discharge_date": discharge_date,
        "social_security_number": social_security_number,
        "tax_identification_number": tax_identification_number,
        "work_permit": work_permit, "own_car": own_car, "datev_id": datev_id,
    }.items() if v is not None}

    _log(caller_username, "update_employee", f"id={employee_id} fields={list(payload.keys())}")
    resp = await _http.patch(f"/employees/{employee_id}", json=payload)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def create_employee(
    caller_username: str,
    first_name: str,
    last_name: str,
    department_id: int,
    title: Optional[str] = None,
    gender: Optional[int] = None,
    birth_date: Optional[str] = None,
    nationality: Optional[str] = None,
    street: Optional[str] = None,
    street_no: Optional[str] = None,
    zipcode: Optional[str] = None,
    place: Optional[str] = None,
    country: Optional[str] = None,
    mobile: Optional[str] = None,
    email: Optional[str] = None,
    account_number: Optional[str] = None,
    employment_id: Optional[str] = None,
    first_entry_date: Optional[str] = None,
    visibility: int = 1,
) -> dict:
    """
    Create a new employee. Requires hr_admin or system_admin role.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        first_name: Employee first name (required).
        last_name: Employee last name (required).
        department_id: Primary department ID (required).
        (all other args): Optional fields.
    """
    if not _is_admin(caller_username):
        return _deny("create", caller_username, "employee records")

    payload = {k: v for k, v in {
        "first_name": first_name, "last_name": last_name,
        "department_id": department_id, "title": title, "gender": gender,
        "birth_date": birth_date, "nationality": nationality,
        "street": street, "street_no": street_no, "zipcode": zipcode,
        "place": place, "country": country, "mobile": mobile, "email": email,
        "account_number": account_number, "employment_id": employment_id,
        "first_entry_date": first_entry_date, "visibility": visibility,
    }.items() if v is not None}

    _log(caller_username, "create_employee", f"name={first_name} {last_name}")
    resp = await _http.post("/employees", json=payload)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def delete_employee(caller_username: str, employee_id: str) -> dict:
    """
    Permanently delete an employee. Requires hr_admin or system_admin role.
    Use check_employee_deletable first.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        employee_id: The employee's numeric ID.
    """
    if not _is_admin(caller_username):
        return _deny("delete", caller_username, f"employee {employee_id}")

    _log(caller_username, "delete_employee", f"id={employee_id}")
    resp = await _http.delete(f"/employees/{employee_id}")
    if resp.status_code == 204:
        return {"success": True, "message": f"Employee {employee_id} deleted."}
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def check_employee_deletable(caller_username: str, employee_id: str) -> dict:
    """
    Check whether an employee can be safely deleted before attempting deletion.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        employee_id: The employee's numeric ID.
    """
    if not _is_admin(caller_username):
        return _deny("check-delete", caller_username, f"employee {employee_id}")

    _log(caller_username, "check_employee_deletable", f"id={employee_id}")
    resp = await _http.get(f"/employees/deletable/{employee_id}")
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def get_employee_summary(caller_username: str, employee_id: str) -> dict:
    """
    Get an employee's summary including vacation and hours overview.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        employee_id: The employee's numeric ID.
    """
    if not _can_read(caller_username, employee_id):
        return _deny("read summary of", caller_username, f"employee {employee_id}")

    _log(caller_username, "get_employee_summary", f"id={employee_id}")
    resp = await _http.get(f"/employees/{employee_id}/summary")
    resp.raise_for_status()
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_projects(caller_username: str, visibility: Optional[int] = None) -> dict:
    """
    Get all projects. Available to all authenticated users.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        visibility: Optional filter — 0=hidden, 1=visible.
    """
    _log(caller_username, "get_projects")
    params = {}
    if visibility is not None:
        params["filter[visibility]"] = visibility
    resp = await _http.get("/projects/range", params=params)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def get_project_by_id(caller_username: str, project_id: str) -> dict:
    """
    Get detailed information about a specific project.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        project_id: The project's ID.
    """
    _log(caller_username, "get_project_by_id", f"id={project_id}")
    resp = await _http.get(f"/projects/range/{project_id}")
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
async def search_projects(caller_username: str, query: str) -> dict:
    """
    Search for projects by name or description.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        query: Search term.
    """
    _log(caller_username, "search_projects", f"query={query}")
    resp = await _http.get(f"/projects/range/search/{query}")
    resp.raise_for_status()
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# TIME BALANCE TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_employee_time_balances(
    caller_username: str,
    employee_id: str,
    range_begin: str,
    range_end: str,
) -> dict:
    """
    Get time balance entries for an employee within a date range.

    Args:
        caller_username: The authenticated username (injected by system prompt).
        employee_id: The employee's numeric ID.
        range_begin: Start date in YYYY-MM-DD format.
        range_end: End date in YYYY-MM-DD format.
    """
    if not _can_read(caller_username, employee_id):
        return _deny("read time balances of", caller_username, f"employee {employee_id}")

    _log(caller_username, "get_employee_time_balances", f"id={employee_id} {range_begin}→{range_end}")
    resp = await _http.get(
        f"/employees/{employee_id}/time_balances",
        params={"filter[range_begin]": range_begin, "filter[range_end]": range_end},
    )
    resp.raise_for_status()
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    # FastMCP exposes a Starlette/ASGI app via .http_app() for SSE transport
    app = mcp.http_app()
    print(f"🚀 SecureHR MCP Server starting on port {MCP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT)