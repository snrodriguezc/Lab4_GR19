# Lab4_GR19

## Despliegue de API en ambiente local
1. Ubicarse en la carpeta raiz del proyecto: ./LAB4 - API
2. Abrir una nueva terminal
3. Correr el servidor con el comando: uvicorn main:app --reload

## Consumo de los endpoints
1. Definir la ruta principal para las peticiones HTTP. En este caso: http://127.0.0.1:8000
2. Agregar el endpoint sobre el cual se quieren enviar las peticiones: /train (entrenar) o /predict (hacer predicciones)
3. Agregar en el body de la petición los registros en formato JSON: 
* [EJEMPLO ENTRENAMIENTO](https://github.com/snrodriguezc/Lab4_GR19/blob/main/Data/university_admission_train.json)
* [EJEMPLO PREDICCIONES](https://github.com/snrodriguezc/Lab4_GR19/blob/main/Data/predict_esc2.json)
5. Verificar que la petición sea de tipo POST y ejecutarla
* [PETICIONES POSTMAN](https://github.com/snrodriguezc/Lab4_GR19/blob/main/Collections/Lab4.postman_collection.json)

## Entregables
* [Script] (https://github.com/snrodriguezc/Lab4_GR19/blob/main/Pipeline.ipynb) de Jupyter Lab con la construcción del pipeline y la explicación de cada uno de los pasos que lo componen.
