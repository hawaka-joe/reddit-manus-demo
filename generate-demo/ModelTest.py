from dataclasses import dataclass

from pydantic import BaseModel


class ModelTest(BaseModel):
    model: str
    prompt: str
    expected_output: str

    def test(self):
        pass
    
if __name__ == "__main__":
    print(ModelTest.model_json_schema())