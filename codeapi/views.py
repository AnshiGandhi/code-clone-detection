# from django.shortcuts import render
# from rest_framework.decorators import api_view
# from rest_framework.response import Response

# # Create your views here.

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .clone_detection_service import CloneDetector
import os

# @api_view(['GET'])
# def hello_world(request):
#     # if request.method == 'POST':
#     #     return Response({"message": "Got some data!", "data": request.data})
#     return Response({"message": "Hello, world!"})




@api_view(['POST'])
def process_request(request):
    """
    API to process the given JSON object and return a response with 'k' objects.
    """
    try:
        # Extract data from the request
        code = request.data.get('code', '')
        candidatesJson = request.data.get('candidates', {})
        k = request.data.get('k', 0)

        # Validate input
        if not isinstance(code, str) or not isinstance(candidatesJson, dict) or not isinstance(k, int):
            return Response(
                {"error": "Invalid input format"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        else:
            keys = list(candidatesJson.keys())
            candidates = list(candidatesJson.values())

        print("Keys:", keys)
        print("Values:", candidates)

        # Your formatting and logic here
        # This example returns the first 'k' keys with dummy integer values.
        # result = {f"keyNum{i+1}": idx for i, (key, idx) in enumerate(candidates.items()) if i < k}

        # clone_detector = CloneDetector("/content/gdrive/MyDrive/final/fine_tuned_model_final")
        clone_detector = CloneDetector(os.path.join( "backend", "final", "fine_tuned_model_final"))



        result = clone_detector.process_candidates(keys, candidates, k, code)

        # Response
        return Response(result, status=status.HTTP_200_OK)

    except Exception as e:
        # Handle unexpected errors
        return Response(
            {"error": f"Something went wrong: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
