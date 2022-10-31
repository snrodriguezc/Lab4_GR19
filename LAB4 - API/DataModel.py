from pydantic import BaseModel

#Representa el esquema de los datos de entrada para el entrenamiento
class DataModelTrain(BaseModel):
    serial_no: float | None = None
    gre_score: float | None = None
    toefl_score: float | None = None
    university_rating: float | None = None
    sop: float | None = None
    lor: float | None = None
    cgpa: float | None = None
    research: float | None = None
    admission_points: float | None = None

    #Representa las columnas del dataframe para el entrenamiento
    def columnsTrain(self):
        return ["Serial No.","GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research","Admission Points"]

#Representa el esquema de los datos de entrada para las predicciones
class DataModelPred(BaseModel):
    serial_no: float | None = None
    gre_score: float | None = None
    toefl_score: float | None = None
    university_rating: float
    sop: float | None = None
    lor: float | None = None
    cgpa: float
    research: float

    #Representa las columnas del dataframe para las predicciones
    def columnsPred(self):
        return ["Serial No.","GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]

#Clase para extraer la lista de registros dados para las predicciones
class ListModelPred(BaseModel):
    registros: list[DataModelPred]

#Clase para extraer la lista de registros dados para el entrenamiento
class ListModelTrain(BaseModel):
    registros: list[DataModelTrain]
