from langchain_core.tools import tool

document_content = ""


def get_document():
    return document_content


@tool
def update(content: str) -> str:
    """Update the document with the provided content."""

    global document_content

    document_content = content

    return (
        "Document has been updated successfully!\n"
        f"The current content is:\n{document_content}"
    )


@tool
def save(filename: str) -> str:
    """Save the current document to a text file."""

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)

        print(f"\n💾 Document has been saved to: {filename}")

        return f"Document has been saved successfully to '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [update, save]
