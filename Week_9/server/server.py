import json
from typing import Any, Dict, List, Optional
from fastmcp import FastMCP, settings

# Configure FastMCP for clean, headless execution in stdio mode
settings.check_for_updates = "off"
settings.show_server_banner = False
import sys
from pathlib import Path

# Ensure directory of server.py is in sys.path for local module resolution
server_dir = str(Path(__file__).resolve().parent)
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

try : 
    from .data import PRODUCTS, EMPLOYEES
except Exception as e:
    print(f"Error importing data: {e}")
    from data import PRODUCTS, EMPLOYEES


# Initialize the FastMCP server instance
mcp = FastMCP(
    name="Enterprise-Catalog-and-Directory",
    instructions="Provides searchable tools and resources for enterprise products, inventory, and employee directory."
)


# ==============================================================================
# MCP RESOURCES
# ==============================================================================

@mcp.resource("catalog://products")
def get_all_products() -> str:
    """Read full list of all products in the catalog (JSON)."""
    return json.dumps(PRODUCTS, indent=2)


@mcp.resource("catalog://categories")
def get_product_categories() -> str:
    """Read summary of available product categories and item counts."""
    categories: Dict[str, int] = {}
    for p in PRODUCTS:
        cat = p.get("category", "Uncategorized")
        categories[cat] = categories.get(cat, 0) + 1
    return json.dumps({
        "categories": categories,
        "total_products": len(PRODUCTS)
    }, indent=2)


@mcp.resource("catalog://product/{product_id}")
def get_product_resource(product_id: str) -> str:
    """
    Dynamic resource template for a specific product by its ID.
    Example URI: catalog://product/PROD-001
    """
    for p in PRODUCTS:
        if p["id"].lower() == product_id.lower():
            return json.dumps(p, indent=2)
    return json.dumps({"error": f"Product with ID '{product_id}' not found."}, indent=2)


@mcp.resource("directory://employees")
def get_all_employees() -> str:
    """Read full company employee directory (JSON)."""
    return json.dumps(EMPLOYEES, indent=2)


@mcp.resource("directory://employee/{employee_id}")
def get_employee_resource(employee_id: str) -> str:
    """
    Dynamic resource template for a specific employee profile.
    Example URI: directory://employee/EMP-101
    """
    for e in EMPLOYEES:
        if e["id"].lower() == employee_id.lower():
            return json.dumps(e, indent=2)
    return json.dumps({"error": f"Employee with ID '{employee_id}' not found."}, indent=2)


# ==============================================================================
# MCP TOOLS
# ==============================================================================

@mcp.tool()
def search_products(
    query: str = "",
    category: str = "",
    max_price: Optional[float] = None,
    in_stock_only: bool = False
) -> Dict[str, Any]:
    """
    Search the product catalog with multi-criteria filtering.

    Args:
        query: Keyword to search in product name or description (e.g. 'headphones', 'desk').
        category: Optional category filter (e.g. 'Electronics', 'Office Furniture').
        max_price: Optional maximum price ceiling.
        in_stock_only: If True, only returns items with stock > 0.

    Returns:
        Dictionary containing matched products and total match count.
    """
    results: List[Dict[str, Any]] = []
    q = query.strip().lower()
    cat = category.strip().lower()

    for item in PRODUCTS:
        # 1. Keyword check
        if q and (q not in item["name"].lower() and q not in item["description"].lower()):
            continue

        # 2. Category check
        if cat and cat not in item["category"].lower():
            continue

        # 3. Max price check
        if max_price is not None and item["price"] > max_price:
            continue

        # 4. Stock check
        if in_stock_only and item["stock"] <= 0:
            continue

        results.append(item)

    return {
        "count": len(results),
        "products": results
    }


@mcp.tool()
def get_product_details(product_id: str) -> Dict[str, Any]:
    """
    Fetch comprehensive details, technical specifications, and current stock for a specific product ID.

    Args:
        product_id: The unique product identifier (e.g. 'PROD-001').
    """
    pid = product_id.strip().upper()
    for item in PRODUCTS:
        if item["id"].upper() == pid:
            return {
                "found": True,
                "product": item
            }
    return {
        "found": False,
        "message": f"Product with ID '{product_id}' was not found in catalog."
    }


@mcp.tool()
def check_inventory(product_id: str) -> Dict[str, Any]:
    """
    Check real-time warehouse inventory and stock availability for a product.

    Args:
        product_id: The unique product identifier (e.g. 'PROD-003').
    """
    pid = product_id.strip().upper()
    for item in PRODUCTS:
        if item["id"].upper() == pid:
            stock = item["stock"]
            status = "In Stock" if stock > 10 else ("Low Stock" if stock > 0 else "Out of Stock")
            return {
                "product_id": item["id"],
                "product_name": item["name"],
                "units_available": stock,
                "status": status,
                "reorder_threshold": 10
            }
    return {
        "product_id": product_id,
        "status": "Unknown",
        "message": f"Product '{product_id}' does not exist in inventory records."
    }


@mcp.tool()
def search_employees(
    query: str = "",
    department: str = "",
    skill: str = ""
) -> Dict[str, Any]:
    """
    Search the company employee directory by name, role, department, or technical skill.

    Args:
        query: Name or role keyword (e.g. 'Aarav', 'Engineer', 'Product Manager').
        department: Department filter (e.g. 'Engineering', 'Product', 'Infrastructure').
        skill: Specific technical skill or domain expertise (e.g. 'Python', 'MCP', 'Kubernetes').
    """
    results: List[Dict[str, Any]] = []
    q = query.strip().lower()
    dept = department.strip().lower()
    sk = skill.strip().lower()

    for emp in EMPLOYEES:
        # 1. Name or Role check
        if q and (q not in emp["name"].lower() and q not in emp["role"].lower()):
            continue

        # 2. Department check
        if dept and dept not in emp["department"].lower():
            continue

        # 3. Skill check
        if sk:
            skills_lower = [s.lower() for s in emp.get("skills", [])]
            if not any(sk in s for s in skills_lower):
                continue

        results.append(emp)

    return {
        "count": len(results),
        "employees": results
    }


@mcp.tool()
def get_employee_details(employee_id: str) -> Dict[str, Any]:
    """
    Retrieve full profile and reporting information for an employee.

    Args:
        employee_id: Unique employee ID (e.g. 'EMP-101').
    """
    eid = employee_id.strip().upper()
    for emp in EMPLOYEES:
        if emp["id"].upper() == eid:
            # Find manager name if manager_id exists
            manager_name = None
            if emp.get("manager_id"):
                manager = next((m for m in EMPLOYEES if m["id"] == emp["manager_id"]), None)
                manager_name = manager["name"] if manager else None

            return {
                "found": True,
                "employee": emp,
                "manager_name": manager_name
            }
    return {
        "found": False,
        "message": f"Employee with ID '{employee_id}' not found in directory."
    }


# ==============================================================================
# MCP PROMPTS
# ==============================================================================

@mcp.prompt()
def compare_products_prompt(product_id_1: str, product_id_2: str) -> str:
    """Generate a structured prompt to compare two products from the catalog."""
    return (
        f"Please perform a detailed comparison between products '{product_id_1}' and '{product_id_2}'.\n"
        f"1. Use the 'get_product_details' tool to fetch specs for both products.\n"
        f"2. Compare their price, specifications, ratings, and stock availability.\n"
        f"3. Provide a clear pros & cons table and a final recommendation for different user personas."
    )


@mcp.prompt()
def route_support_ticket_prompt(product_id: str, issue_description: str) -> str:
    """Generate a support routing prompt to match a customer issue with the right team."""
    return (
        f"A customer reported the following issue with product '{product_id}':\n"
        f"\"{issue_description}\"\n\n"
        f"Please:\n"
        f"1. Check the product details using 'get_product_details'.\n"
        f"2. Search the employee directory using 'search_employees' to find the most appropriate "
        f"person (by department or skill) to handle this issue.\n"
        f"3. Draft a courteous response to the customer with an internal ticket assignment note."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
