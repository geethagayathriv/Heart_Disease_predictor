from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
import pickle
import numpy as np

app=FastAPI()
app.mount("/static",StaticFiles(directory="static"), name="static")
templates=Jinja2Templates(directory="templates")

@app.get("/",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("disease_prediction.html", {"request": request})

@app.get("/predictor.html",response_class=HTMLResponse)
async def predictor(request: Request):
    return templates.TemplateResponse("predictor.html", {"request": request})

@app.get("/redirect")
async def redirect_to_predictor():
    return RedirectResponse(url="/predictor.html")

with open("model.pkl","rb") as file:
    model=pickle.load(file)

@app.post("/predict",response_class=HTMLResponse)
async def predict(request: Request,
                  highbp: int=Form(...),
                  cholestrol: int = Form(...),
                  chol5: int = Form(...),
                  bmi: float = Form(...),
                  smoke: int = Form(...),
                  heart: int = Form(...),
                  diabetes: int = Form(...),
                  active: int = Form(...),
                  fruits: int = Form(...),
                  vegetables: int = Form(...),
                  alcohol: int = Form(...),
                  healthcare: int = Form(...),
                  no_doc_visit: int = Form(...),
                  general_health: int = Form(...),
                  mental_health: int = Form(...),
                  physical_health: int = Form(...),
                  stairs: int = Form(...),
                  sex: int=Form(...),
                  age: int = Form(...)):
    data=np.array([[highbp,cholestrol,chol5,bmi,smoke,heart,diabetes,active,fruits,vegetables,alcohol,healthcare,no_doc_visit,general_health,mental_health,physical_health,stairs,sex,age]])
    prediction=model.predict(data)[0]
    if prediction==1.0:
        return templates.TemplateResponse("results.html", {"request": request, "prediction": "You have a chance of incidence, consider visiting a doctor"})
    if prediction==0.0:
        return templates.TemplateResponse("results.html", {"request": request, "prediction": "Your heart is healthy"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)