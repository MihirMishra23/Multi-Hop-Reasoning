from pydantic import BaseModel
from lmlm.constants import DB_END_TOKEN, DB_RETRIEVE_TOKEN, DB_SEP_TOKEN, DB_START_TOKEN,ANSWER_START_TOKEN, ANSWER_END_TOKEN, THINKING_START_TOKEN, THINKING_END_TOKEN

class RolloutMetadata(BaseModel):
    question : str
    full_response : str
    annotated_text : str
    triplets : str
    golden_answer : list[str]
    f1_score: float
    lmlm_answer : str | None
    input_tokens: int | None = None
    output_tokens: int | None = None



def is_valid_rollout(text: str) -> bool:
    if text.count(ANSWER_START_TOKEN) != 1:
        print(f"Invalid rollout: Must have exactly one ANSWER_START_TOKEN, found {text.count(ANSWER_START_TOKEN)}")
        return False
    if text.count(ANSWER_END_TOKEN) != 1:
        print(f"Invalid rollout: Must have exactly one ANSWER_END_TOKEN, found {text.count(ANSWER_END_TOKEN)}")
        return False
    if text.count(THINKING_START_TOKEN) != 1:
        print(f"Invalid rollout: Must have exactly one {THINKING_START_TOKEN} tag, found {text.count(THINKING_START_TOKEN)}")
        return False
    if text.count(THINKING_END_TOKEN) != 1:
        print(f"Invalid rollout: Must have exactly one {THINKING_END_TOKEN} tag, found {text.count(THINKING_END_TOKEN)}")
        return False
    if text.count(DB_START_TOKEN) < 1:
        print(f"Invalid rollout: Must have at least one DB_START_TOKEN, found {text.count(DB_START_TOKEN)}")
        return False
    if text.count(DB_END_TOKEN) < 1:
        print(f"Invalid rollout: Must have at least one DB_END_TOKEN, found {text.count(DB_END_TOKEN)}")
        return False
    if text.count(DB_SEP_TOKEN) < 1:
        print(f"Invalid rollout: Must have at least one DB_SEP_TOKEN, found {text.count(DB_SEP_TOKEN)}")
        return False
    if text.count(DB_RETRIEVE_TOKEN) < 1:
        print(f"Invalid rollout: Must have at least one DB_RETRIEVE_TOKEN, found {text.count(DB_RETRIEVE_TOKEN)}")
        return False
    return True