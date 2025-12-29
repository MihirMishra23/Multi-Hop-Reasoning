from pydantic import BaseModel
from lmlm.constants import DB_END_TOKEN, DB_RETRIEVE_TOKEN, DB_SEP_TOKEN, DB_START_TOKEN,ANSWER_START_TOKEN, ANSWER_END_TOKEN, THINKING_START_TOKEN, THINKING_END_TOKEN

class RolloutMetadata(BaseModel):
    question : str
    full_response : str
    annotated_text : str
    triplets : str
    golden_answer : list[str]
    f1_score: float
    lmlm_answer : str



def assert_valid_rollout(text: str):
    assert text.count(ANSWER_START_TOKEN) == 1, "Must have exactly one ANSWER_START_TOKEN"
    assert text.count(ANSWER_END_TOKEN) == 1, "Must have exactly one ANSWER_END_TOKEN"
    assert text.count(THINKING_START_TOKEN) == 1, f"Must have exactly one {THINKING_START_TOKEN} tag"
    assert text.count(THINKING_END_TOKEN) == 1, f"Must have exactly one {THINKING_END_TOKEN} tag"
    assert text.count(DB_START_TOKEN) >= 1, "Must have at least one DB_START_TOKEN"
    assert text.count(DB_END_TOKEN) >= 1, "Must have at least one DB_END_TOKEN"
    assert text.count(DB_SEP_TOKEN) >= 1, "Must have at least one DB_SEP_TOKEN"
    assert text.count(DB_RETRIEVE_TOKEN) >= 1, "Must have at least one DB_RETRIEVE_TOKEN"
    return True