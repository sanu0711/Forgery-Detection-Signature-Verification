from django.shortcuts import render, redirect

# Create your views here.


def homeView(request):
    return render(request, 'home.html')

def signatureVerificationView(request):
    if request.method == 'POST':
        signature_img = request.FILES.get('signature')
        print(signature_img)
        return render(request, 'signatureVerification.html',{"isverified":1})
    # return redirect('home')
    SequentialModel = helperformatModelOutput("Sequential Model", "Sequential", 0.95, 0.85)
    resnet50Model = helperformatModelOutput("ResNet50 Model", "ResNet50", 0.95, 0.85)
    vgg16Model = helperformatModelOutput("VGG16 Model", "VGG16", 0.95, 0.85)
    vgg19Model = helperformatModelOutput("VGG19 Model", "VGG19", 0.95, 0.85)
    context = {
        "real_conf": 85.0,
        "fake_conf": 15.0,
        "avg_model_accuracy": 95.6,
        
        "SequentialModel": SequentialModel,
        "resnet50Model": resnet50Model,
        "vgg16Model": vgg16Model,
        "vgg19Model": vgg19Model,
        "isverified":0,
        
    }
    return render(request, 'signatureVerification.html', context)

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
    
        
    
    