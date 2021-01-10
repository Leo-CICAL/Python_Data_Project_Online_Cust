from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
model = pickle.load(open('rf_customer.pkl', 'rb'))


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    Administrative = request.form['a']
    Administrative_Duration  = request.form['b']
    Informational = request.form['c']
    Informational_Duration = request.form['d']
    ProductRelated = request.form['e']
    ProductRelated_Duration = request.form['f']
    BounceRates = request.form['g']
    ExitRates = request.form['h']
    PageValues = request.form['i']
    SpecialDay = request.form['j']
    Month = request.form['k']
    OperatingSystems = request.form['l']
    Browser = request.form['m']
    Region = request.form['n']
    TrafficType = request.form['o']
    VisitorType = request.form['p']
    Weekend = request.form['q']
    month = ['August','December','February','July','June','March','May','November','October','September']
    visitortype = ['New_Visitor','Other','Returning_Visitor']
    if Month not in month:
        return render_template('home.html', msg='The Month is not good ! It should be August, December, February, July, June , March, May, November, October or September.')
    if VisitorType not in visitortype:
        return render_template('home.html', msg='The VistorType is not good ! It should be New_Visitor, Other, Returning_Visitor')
    arr_month = [0,0,0,0,0,0,0,0,0,0]
    index_month = month.index(Month)
    arr_month[index_month] = 1
    arr_visitortype = [0,0,0]
    index_visitor = visitortype.index(VisitorType)
    arr_visitortype[index_visitor] = 1
    arr_browser = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    arr_browser[int(Browser) - 1] = 1
    arr_operating = [0,0,0,0,0,0,0,0]
    arr_operating[int(OperatingSystems) - 1] = 1
    arr_region = [0,0,0,0,0,0,0,0,0]
    arr_region[int(Region) - 1] = 1
    arr_traffic = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    arr_traffic[int(TrafficType) - 1] = 1
    arr_weekend = [0,0]
    arr_weekend[int(Weekend) -1] = 1
    arr = np.array([[float(Administrative) , float(Administrative_Duration), float(Informational), float(Informational_Duration),float(ProductRelated),float(ProductRelated_Duration),float(BounceRates),float(ExitRates),float(PageValues),float(SpecialDay),
     arr_month[0], arr_month[1], arr_month[2], arr_month[3], arr_month[4], arr_month[5], arr_month[6], arr_month[7], arr_month[8], arr_month[9],
     arr_operating[0],arr_operating[1],arr_operating[2],arr_operating[3],arr_operating[4],arr_operating[5],arr_operating[6],arr_operating[7],
     arr_browser[0],arr_browser[1],arr_browser[2],arr_browser[3],arr_browser[4],arr_browser[5],arr_browser[6],arr_browser[7],arr_browser[8],arr_browser[9],arr_browser[10],arr_browser[11],arr_browser[12],
     arr_region[0],arr_region[1],arr_region[2],arr_region[3],arr_region[4],arr_region[5],arr_region[6],arr_region[7],arr_region[8],
     arr_traffic[0],arr_traffic[1],arr_traffic[2],arr_traffic[3],arr_traffic[4],arr_traffic[5],arr_traffic[6],arr_traffic[7],arr_traffic[8],arr_traffic[9],
     arr_traffic[10],arr_traffic[11],arr_traffic[12],arr_traffic[13],arr_traffic[14],arr_traffic[15],arr_traffic[16],arr_traffic[17],arr_traffic[18],arr_traffic[19],
     arr_visitortype[0],arr_visitortype[1],arr_visitortype[2],arr_weekend[0],arr_weekend[1]]])
    
    scaler = StandardScaler()
    arr_scaled = StandardScaler().fit_transform(arr)
    pred = model.predict(arr_scaled)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)