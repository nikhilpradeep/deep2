import sys,os,shutil
sys.path.append(os.getcwd())
from src.predict import *;
from src.model.nets import resnet
from src.model.transforms import tf
from flask import Flask, request, flash, redirect, render_template, url_for, Markup, send_file,send_from_directory
from werkzeug.utils import secure_filename
from src.model.eda import eda
from threading import Timer
import webbrowser

app = Flask(__name__)
app.secret_key = "!@#$%^&*()a-=afs;'';312$%^&*k-[;.sda,./][p;/'=-0989#$%^&0976678v$%^&*(fdsd21234266OJ^&UOKN4odsbd#$%^&*(sadg7(*&^%32b342gd']"
# the upload path for all the files

SUB_UPLOAD_FOLDER = "static/uploadFolder"
imagefolder = "static/image"
UPLOAD_FOLDER ="flask_api/"+ SUB_UPLOAD_FOLDER
UPLOAD_FOLDER2 = "flask_api/"+imagefolder
# a list to track all the files loaded in memory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
list_of_uploaded_file = []

model = resnet.ResNet18()
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def clean_upload_folder():
    try:
        shutil.rmtree(UPLOAD_FOLDER + "/")
        shutil.rmtree(UPLOAD_FOLDER2 + "/")
    except FileNotFoundError as e:
        pass


def make_directory():
    os.makedirs(UPLOAD_FOLDER,exist_ok=True)
   
    
def check_checkpoint():
    
    
    if os.path.exists('.\src\checkpoint\ckpt.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        state_dict = torch.load('.\src\checkpoint\ckpt.pth',map_location=device)['net']
        newdict = {}
        for k, v in state_dict.items():
            newdict[k[7:]] = v
		# load params
        model.load_state_dict(newdict)
        model.eval()
    else:
        return "Checkpoint file not found. Please train."

@app.route("/goback", methods=['POST'])
def goback():
    if request.method == 'POST':
        
        return redirect('/')

@app.route('/', methods=['GET', 'POST'])
def root():
    # clean the upload directory every time user use the website and create a new empty directory
    clean_upload_folder()
    make_directory()
    
    results = eda.eda()
    return render_template("upload.html",trl=results[0],tel=results[1],cls=results[2],ls=results[3],cls1=results[4],pathi=os.path.join(imagefolder,"cifar10.png"))
    

@app.route('/train', methods=['GET', 'POST'])
def training():
    t()
    return 'the model has trained'
    
@app.route('/random_t', methods=['GET', 'POST'])
def random_test():
    check_checkpoint()
    if os.path.exists('src\\data\\cifar-10-batches-py\\test_batch'):
        testimgs=unpickle('src\\data\\cifar-10-batches-py\\test_batch')
        testimgs=testimgs[b'data']
	
    else:
        return "Test data doesnt exists. Please check."
    result = r(testimgs,model,tf)
    return render_template("result.html",result=result,image_path = os.path.join(imagefolder,"res.png"))

@app.route("/upload",methods=["POST","GET"])
def upload_image_file():
    check_checkpoint()
    print("root testing" , request.files)
    if request.method == "POST":
        # check wether the request value is 
        print("upload file" , request.files)
        if "file" in request.files:
            # get the multiDict
            file = request.files.getlist("file")[0]
            print("upload file" , request.files)
            # secure the filename which will only give file name excluding other parameters
            filename = secure_filename(file.filename)
            #print(filename , "  " , dir(file.stream))
            # get the file path
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            print(filename)
            print(path)
            result  = u(model,tf,path)
            return render_template("result.html",result=result,image_path = os.path.join(SUB_UPLOAD_FOLDER,filename))


@app.route("/test")
def test():
    return "Hello World"

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__  =="__main__":
    Timer(2,webbrowser.open('http://127.0.0.1:5000/'))
    app.run(debug=False, host="127.0.0.1", port=5000)
