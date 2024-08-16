import json
from datetime import datetime
import pytz
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

def get_time(location: str = "utc"):
    """Get the current time in the specified time zone."""
    try:
        # Attempt to set the time zone
        tz = pytz.timezone(location)
    except pytz.UnknownTimeZoneError:
        # If the time zone is not recognized, return an error message
        return json.dumps({"error": f"Unknown time zone: {location}"})

    # Get the current time in the specified time zone
    now = datetime.now(tz)
    # Format the time as YYYY/MM/DD hh:mm:ss
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    return json.dumps({f"{location} time": f"{current_time}"})

def RAG_mixture_of_agents(message):
    """Search informations about mixture of agents."""
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PATH = "chroma"
    embedding_function = get_embedding_function(model_name=embed_model)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(message, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    return json.dumps({f"Result": f"{context_text}"})
    