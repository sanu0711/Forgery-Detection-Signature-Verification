from django.shortcuts import render, redirect
from .model_prediction import prediction_score
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Create your views here.

SequentialModel_accuracy = 96.20
resnet50Model_accuracy = 79.20
vgg16Model_accuracy = 92.40
vgg19Model_accuracy = 93.00


def homeView(request):
    return render(request, 'home.html')

def signatureVerificationView(request):
    if request.method == 'POST':
        signature_img = request.FILES.get('signature')
        fs = FileSystemStorage()
        filename = fs.save(f'signatures/{signature_img.name}', signature_img)  # saved to media/signatures/
        image_url = fs.url(filename)  # resolves to /media/signatures/filename

        # Use the saved file path for prediction if needed
        # saved_path = os.path.join(settings.MEDIA_ROOT, filename)
        # print(signature_img)
        sequential = prediction_score('forg.h5', signature_img, (150, 150))
        resnet50 = prediction_score('resnet_model.h5', signature_img, ( 64, 64))
        vgg16 = prediction_score('vgg16_model.h5', signature_img,(64, 64)) 
        vgg19 = prediction_score('vgg19_model.h5', signature_img,(64, 64)) 
        # vgg19 = 0.12
        SequentialModel = helperformatModelOutput("Sequential Model", "Sequential",SequentialModel_accuracy, sequential)
        resnet50Model = helperformatModelOutput("ResNet50 Model", "ResNet50", resnet50Model_accuracy, resnet50)
        vgg16Model = helperformatModelOutput("VGG16 Model", "VGG16", vgg16Model_accuracy, vgg16)
        vgg19Model = helperformatModelOutput("VGG19 Model", "VGG19", vgg19Model_accuracy, vgg19)
        avg_model_accuracy, real_conf, fake_conf, isverified = helperOverallModelOutput(sequential,resnet50,vgg16,vgg19)
        context = {
            "real_conf": real_conf,
            "fake_conf": fake_conf,
            "avg_model_accuracy": avg_model_accuracy,
            "isverified": isverified,
            
            "SequentialModel": SequentialModel,
            "resnet50Model": resnet50Model,
            "vgg16Model": vgg16Model,
            "vgg19Model": vgg19Model,
            "signature_image_url": image_url
        }
        return render(request, 'signatureVerification.html', context)
    return redirect('home')
    

def helperOverallModelOutput(sequential,resnet50,vgg16,vgg19):
    avg_model_accuracy = (SequentialModel_accuracy + resnet50Model_accuracy + vgg16Model_accuracy + vgg19Model_accuracy) / 4
    # real_conf = round((sequential + resnet50 + vgg16 + vgg19)*100 / 4, 2)
    real_conf = round((sequential + resnet50 + vgg19)*100 / 3, 2)
    fake_conf = 100 - real_conf
    if real_conf > 50:
        isverified = 1
    elif real_conf <= 50:
        isverified = 0
    else:
        isverified = -1
    return avg_model_accuracy, real_conf, fake_conf, isverified
    
def helperformatModelOutput(model_name,functional,accuracy,confidence):
    if confidence > 0.5:
        isverified = 1
    elif confidence <= 0.5:
        isverified = 0
    else:
        isverified = -1
    print(isverified)
    real_conf = round(confidence * 100, 2)
    fake_conf = round((1 - confidence) * 100, 2)
    return {
        "model_name": model_name,
        "functional": functional,
        "accuracy": accuracy,
        "real_conf": real_conf,
        "fake_conf": fake_conf,
        "isverified": isverified
    }
    
        
    
    