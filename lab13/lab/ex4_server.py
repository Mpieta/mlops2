import base64
import io
import matplotlib.pyplot as plt
from typing import Annotated, List, Optional
from fastmcp import FastMCP

mcp = FastMCP("Visualization Service")


@mcp.tool(description="Create a line plot from data and return it as a base64 encoded PNG image.")
def line_plot(
        data: Annotated[List[float], "A list of numerical values to plot."],
        title: Annotated[Optional[str], "The title of the plot."] = "Line Plot",
        x_label: Annotated[Optional[str], "Label for the X-axis."] = "Index",
        y_label: Annotated[Optional[str], "Label for the Y-axis."] = "Value",
        legend: Annotated[bool, "Whether to show the legend."] = False
) -> str:
    plt.clf()

    plt.plot(data, label=y_label if legend else None)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if legend:
        plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str


if __name__ == "__main__":
    print("Starting Visualization MCP Server on port 8003")
    mcp.run(transport="streamable-http", port=8003)