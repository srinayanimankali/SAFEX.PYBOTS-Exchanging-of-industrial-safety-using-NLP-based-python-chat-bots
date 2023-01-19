from flask import Flask, render_template, request,jsonify


app = Flask(__name__)


import safexpybot

def chatbot_qna(user_response):
    
    if(user_response!='bye') and (user_response!='exit') and (user_response!='quit'):
        if(user_response=='thanks' or user_response=='thank you' ):
            
            return str('Exited from Question and Answer Module')
        else:
            return str(safexpybot.response(user_response))
    elif(user_response=='bye') or (user_response=='exit') or (user_response=='quit'):   
        return str('Bye, Exited from Question and Answer Module')



def chatbot_acclevel(user_response):
    
    if(user_response!='bye') and (user_response!='exit') and (user_response!='quit'):
        if(user_response=='thanks' or user_response=='thank you' ):
            
            return str('Exited from Question and Answer Module')
        else:
            result = safexpybot.response_acclevel(user_response)
            return str('Accident Level is : ' + result[0])
    elif(user_response=='bye') or (user_response=='exit') or (user_response=='quit'):   
        return str('Bye, Exited from Question and Answer Module')        
                
def chatbot_potacclevel(user_response):
    
    if(user_response!='bye') and (user_response!='exit') and (user_response!='quit'):
        if(user_response=='thanks' or user_response=='thank you' ):
            
            return str('Exited from Question and Answer Module')
        else:
            result = safexpybot.response_potacclevel(user_response)
            return str('Potential Accident Level is : ' + result[0])
    elif(user_response=='bye') or (user_response=='exit') or (user_response=='quit'):   
        return str('Bye, Exited from Question and Answer Module')     

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=['POST'])
def ask():
    
    user_response = str(request.form['messageText'])
    user_response=user_response.lower()

    result=chatbot_potacclevel(user_response)
    return jsonify({'status':'OK','answer':result})




if __name__ == "__main__":
    app.run()

