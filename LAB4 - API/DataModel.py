from pydantic import BaseModel

class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    serial_no: float
    gre_score: float
    toefl_score: float
    university_rating: float
    sop: float
    lor: float 
    cgpa: float
    research: float
    admission_points: float

    def columns(self):
        return ["Serial No.","GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research","Admission Points"]

class DataModelPred(BaseModel):
    serial_no: float
    gre_score: float
    toefl_score: float
    university_rating: float
    sop: float
    lor: float 
    cgpa: float
    research: float

    def columnsPred(self):
        return ["Serial No.","GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]

class ListModel(BaseModel):
    registros: list[DataModelPred]
