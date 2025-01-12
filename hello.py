from flask import Flask #import flask class
app = Flask(__name__) #instance of class flask 
@app.route("/") #route decorator tells flask what URL triggers the function

def hello_world():
    return "<p>Upload your mammogram to get a BIRAD score</p>" #default content typre is HTML
# default = only acessibel from this computer. to make it public, add --host = 0.0.0.0 to commmand line
# to enable debug mode (server reloads if code changes), run as: 'flask --app hello run --debug'

# run the code and type 'flask --app hello run' (make sure fals is installed), then you'll get a 'http..' output which is the link 