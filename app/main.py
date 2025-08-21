from fastapi import FastAPI

akiliAi = FastAPI()


@akiliAi.get("/ping")
def first_func():
    return {"message": "AI Engine Ready"}
