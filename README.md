# Code-clone-detection

This project demonstrates semantic analysis (for C/C++). It provides the top 'K' code snippets from a given file (containing different code snippets) that are similar to the input code snippet, along with their similarity scores. We have to ensure that given file should be in JSON format and follows below structure : 
```
{
    "key1" : "code1" ,
    "key2" : "code2" ,
    ....
} 
```



## Virtual environment 

For backend , we have to start virtual environment by using following command : 

```
test\Scripts\activate
```

## Installation

The dependencies can be installed using the following command (requirements file) :

```
pip install -r .\requirements.txt
```

## Running the project backend

To run the project backend, navigate to the parent folder (code-clone-detection) where we have stored file "manage.py" and run following command :
```
python manage.py runserver
```
After successfully executing above commands , backend will be available at url :

```
http://127.0.0.1:8000/process/
```

## Running the project frontend

To run project frontend, navigate to the folder "ccd-ui" and execute following commands  : 

```
npm install
npm run dev
```
After successfully executing above command , frontend will be available at url :
```
http://localhost:5173/
```

