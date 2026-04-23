"""
SecureHR MCP Server - Remote MCP (HTTP/SSE) with RBAC
Exposes Timecount tools to Claude.ai via Model Context Protocol.

Auth: OAuth 2.1 bearer tokens (JWT) in Authorization header
RBAC: Enforced inside every tool based on authenticated user identity
Run:  python mcp_server.py  (listens on :8000, mount path /mcp)
Connect from Claude.ai: Settings -> Connectors -> Add custom MCP server
"""

import os
import re
import json
import httpx
from datetime import datetime
from typing import Optional, Any
from contextvars import ContextVar

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

# ============== Config ==============
TIMECOUNT_API_URL = os.getenv("TIMECOUNT_API_URL", "https://tutorial.formatgold.de/api")
TIMECOUNT_API_TOKEN = os.getenv("TIMECOUNT_API_TOKEN")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET not set")

# Same user format as before: username:password:employee_id:role:department_id
def parse_users():
    users_str = os.getenv("CHAT_USERS", "")
    if not users_str:
        raise RuntimeError("CHAT_USERS not set")
    users = {}
    for row in users_str.split(","):
        parts = row.split(":")
        if len(parts) >= 2:
            users[parts[0].strip()] = {
                "employee_id": parts[2].strip() if len(parts) > 2 and parts[2].strip() else None,
                "role": parts[3].strip() if len(parts) > 3 and parts[3].strip() else "employee",
                "department_id": int(parts[4].strip()) if len(parts) > 4 and parts[4].strip() else None,
            }
    return users

USERS_DB = parse_users()

# ============== Per-request auth context ==============
# MCP tools are called in request scope — we stash the authenticated user here
current_user_ctx: ContextVar[Optional[dict]] = ContextVar("current_user", default=None)

def get_user() -> dict:
    """Return {username, employee_id, role, department_id} for the caller."""
    user = current_user_ctx.get()
    if not user:
        raise PermissionError("Not authenticated")
    return user

# ============== Audit log ==============
def audit(event: str, details: str, status: str, username: str = "system"):
    line = f"{datetime.utcnow().isoformat()} | USER={username} | {event} | {details} | {status}\n"
    try:
        with open("security_audit.log", "a") as f:
            f.write(line)
    except Exception:
        pass
    print(f"🔒 [{status}] {event} - {details}")

# ============== RBAC ==============
def can_access(target_employee_id: str) -> bool:
    u = get_user()
    if u["role"] in ("system_admin", "hr_admin"):
        return True
    return str(u["employee_id"]) == str(target_employee_id)

def can_admin() -> bool:
    return get_user()["role"] in ("system_admin", "hr_admin")

def deny(resource: str, action: str):
    u = get_user()
    audit("ACCESS_DENIED", f"{action} on {resource}", "BLOCKED", u["username"])
    raise PermissionError(
        f"Access denied. You can only {action} your own data (employee_id={u['employee_id']})."
    )

# ============== Prompt injection filter for string inputs ==============
INJECTION_PATTERNS = [
    "ignore previous instructions", "ignore all previous", "disregard previous",
    "new instructions:", "system prompt", "jailbreak", "you are now",
    "forget everything", "pretend you are", "bypass security", "disable safety",
]

def check_injection(value: str, field: str):
    if not isinstance(value, str):
        return
    low = value.lower()
    for p in INJECTION_PATTERNS:
        if p in low:
            u = get_user()
            audit("PROMPT_INJECTION", f"field={field} pattern={p}", "BLOCKED", u["username"])
            raise ValueError(f"Suspicious content in field '{field}'. Please rephrase.")

# ============== Timecount HTTP client ==============
tc = httpx.AsyncClient(
    base_url=TIMECOUNT_API_URL,
    headers={"Authorization": f"Bearer {TIMECOUNT_API_TOKEN}"},
    timeout=30.0,
)

# ============== MCP server ==============
mcp = FastMCP("SecureHR-Timecount")

# ---------- READ tools ----------

@mcp.tool()
async def get_employee_by_id(employee_id: str) -> dict:
    """Get detailed information about a specific employee. Employees can only fetch their own record; admins can fetch anyone."""
    if not can_access(employee_id):
        deny(f"employee:{employee_id}", "view")
    r = await tc.get(f"/employees/{employee_id}")
    r.raise_for_status()
    audit("READ_EMPLOYEE", f"id={employee_id}", "SUCCESS", get_user()["username"])
    return r.json()

@mcp.tool()
async def get_my_employee() -> dict:
    """Get the current authenticated user's employee record. Convenience tool — no ID needed."""
    u = get_user()
    if not u["employee_id"]:
        raise ValueError("Your account is not linked to an employee record.")
    r = await tc.get(f"/employees/{u['employee_id']}")
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def search_employees(query: str) -> dict:
    """Search employees by name or token. Non-admins only see themselves in results."""
    check_injection(query, "query")
    u = get_user()
    if not can_admin():
        # filter: only return self if matches
        if not u["employee_id"]:
            return {"data": []}
        r = await tc.get(f"/employees/{u['employee_id']}")
        r.raise_for_status()
        emp = r.json()
        name = f"{emp.get('first_name','')} {emp.get('last_name','')}".lower()
        if query.lower() in name:
            return {"data": [emp]}
        return {"data": []}
    r = await tc.get(f"/employees/search/{query}")
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def list_employees(visibility: str = "all") -> dict:
    """List all employees. Admin-only; non-admins get their own record only."""
    u = get_user()
    if not can_admin():
        if not u["employee_id"]:
            return {"data": []}
        r = await tc.get(f"/employees/{u['employee_id']}")
        r.raise_for_status()
        return {"data": [r.json()]}
    r = await tc.get("/employees", params={"filter[employee_visibility]": visibility})
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def get_employee_summary(employee_id: str) -> dict:
    """Get vacation/hours summary for an employee. RBAC-enforced."""
    if not can_access(employee_id):
        deny(f"employee:{employee_id}/summary", "view")
    r = await tc.get(f"/employees/{employee_id}/summary")
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def get_employee_time_balances(employee_id: str, range_begin: str, range_end: str) -> dict:
    """Get time balances within a date range (YYYY-MM-DD). RBAC-enforced."""
    if not can_access(employee_id):
        deny(f"employee:{employee_id}/time_balances", "view")
    r = await tc.get(
        f"/employees/{employee_id}/time_balances",
        params={"filter[range_begin]": range_begin, "filter[range_end]": range_end},
    )
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def get_monthly_hours() -> dict:
    """Monthly hours aggregate across all employees. Admin-only."""
    if not can_admin():
        deny("monthly_hours", "view")
    r = await tc.get("/employees/months")
    r.raise_for_status()
    return r.json()

# ---------- WRITE tools ----------

@mcp.tool()
async def update_employee(employee_id: str, fields: dict) -> dict:
    """Update employee fields (PATCH). Pass any subset of: first_name, last_name, street, street_no,
    zipcode, place, country, mobile, email, civil_state_id, account_number, bank_id, etc.
    RBAC: employees can only edit their own record; admins can edit anyone."""
    if not can_access(employee_id):
        deny(f"employee:{employee_id}", "edit")
    # scrub strings for injection
    for k, v in fields.items():
        if isinstance(v, str):
            check_injection(v, k)
    data = {k: v for k, v in fields.items() if v is not None}
    r = await tc.patch(f"/employees/{employee_id}", json=data)
    r.raise_for_status()
    audit("EMPLOYEE_UPDATED", f"id={employee_id} fields={list(data.keys())}", "SUCCESS", get_user()["username"])
    return r.json()

@mcp.tool()
async def create_employee(fields: dict) -> dict:
    """Create a new employee. Admin-only. Requires at least first_name, last_name, department_id."""
    if not can_admin():
        deny("employee:new", "create")
    for k, v in fields.items():
        if isinstance(v, str):
            check_injection(v, k)
    data = {k: v for k, v in fields.items() if v is not None}
    r = await tc.post("/employees", json=data)
    r.raise_for_status()
    result = r.json()
    audit("EMPLOYEE_CREATED", f"id={result.get('id')}", "SUCCESS", get_user()["username"])
    return result

@mcp.tool()
async def delete_employee(employee_id: str) -> dict:
    """Delete an employee permanently. Admin-only."""
    if not can_admin():
        deny(f"employee:{employee_id}", "delete")
    r = await tc.delete(f"/employees/{employee_id}")
    if r.status_code == 204:
        audit("EMPLOYEE_DELETED", f"id={employee_id}", "SUCCESS", get_user()["username"])
        return {"success": True, "message": f"Employee {employee_id} deleted"}
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def check_employee_deletable(employee_id: str) -> dict:
    """Check whether an employee can be safely deleted. Admin-only."""
    if not can_admin():
        deny(f"employee:{employee_id}", "check_deletable")
    r = await tc.get(f"/employees/deletable/{employee_id}")
    r.raise_for_status()
    return r.json()

# ---------- PROJECTS ----------

@mcp.tool()
async def list_projects(visibility: Optional[int] = None) -> dict:
    """List projects. Accessible to all authenticated users."""
    params = {"filter[visibility]": visibility} if visibility is not None else {}
    r = await tc.get("/projects/range", params=params)
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def get_project_by_id(project_id: str) -> dict:
    """Get a project by ID."""
    r = await tc.get(f"/projects/range/{project_id}")
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def search_projects(query: str) -> dict:
    """Search projects."""
    check_injection(query, "query")
    r = await tc.get(f"/projects/range/search/{query}")
    r.raise_for_status()
    return r.json()

# ============== Auth middleware ==============
# Validates JWT Bearer token on every MCP request and populates the context var.

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow unauthenticated access to discovery / health
        if request.url.path in ("/health",) or request.url.path.startswith("/.well-known"):
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"error": "missing bearer token"}, status_code=401)

        token = auth[7:]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            username = payload.get("sub")
            if not username or username not in USERS_DB:
                return JSONResponse({"error": "invalid token"}, status_code=401)
            if payload.get("type") == "temp":
                return JSONResponse({"error": "2FA not completed"}, status_code=401)
        except JWTError:
            return JSONResponse({"error": "invalid token"}, status_code=401)

        user_record = USERS_DB[username]
        token_ctx = current_user_ctx.set({"username": username, **user_record})
        try:
            response = await call_next(request)
        finally:
            current_user_ctx.reset(token_ctx)
        return response

# ============== Mount as ASGI app ==============
app = mcp.streamable_http_app()
app.add_middleware(JWTAuthMiddleware)

@app.route("/health")
async def health(request):
    return JSONResponse({"status": "ok", "users": list(USERS_DB.keys())})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 SecureHR MCP server on :{port}/mcp")
    uvicorn.run(app, host="0.0.0.0", port=port)
