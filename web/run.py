import sys
import pandas as pd
import numpy as np
from flask import Flask, make_response
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, DecimalField,DateField, IntegerField
from wtforms.validators import DataRequired
from config import Config
from flask import render_template, flash, redirect, json, Response, request
from as_detect_helper import select_as
import os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import datetime as dt
from dateutil.relativedelta import relativedelta

here=os.path.dirname(__file__)

asn_df=pd.read_csv(here+"/data/asn.csv",index_col="ASNumber" )
core_df=pd.read_csv(here+"/data/core_norm7.csv",index_col="month")

app = Flask(__name__)
app.config.from_object(Config)
months=[('1','January'),('2','February'),('3','March'),('4','April'),('5','May'),('6','June'),('7','July'),('8','August'),('9','September'),('10','October'),('11','November'),('12','December')]
years=[[str(i), str(i)] for i in range(1998,2018)]

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class SearchAS(FlaskForm):
    from_threshold= StringField('Threshold', description='hola mundo',default=0.3, render_kw={"style":"width:20%","class":"form-control","maxlength":"5"})
    to_threshold= StringField('Threshold', description='hola mundo',default=0.9,render_kw={"style":"width:20%","class":"form-control","maxlength":"5"}) 
    from_month = SelectField('month:',choices=months,default='9',render_kw={"class":"btn btn-default dropdown-toggle"})
    from_year = SelectField('year', choices=years,default='2007',render_kw={"class":"btn btn-default dropdown-toggle"})
    to_month = SelectField('month:',choices=months,default='7',render_kw={"class":"btn btn-default dropdown-toggle"})
    to_year = SelectField('year', choices=years,default='2011',render_kw={"class":"btn btn-default dropdown-toggle"})
    maximum_months= IntegerField('Maximum months of grow',default=12,render_kw={"style":"width:20%","class":"form-control","maxlength":"5"})
    submit=SubmitField('Submit',render_kw={"class":"btn btn-primary"})

@app.route('/', methods=['GET', 'POST'])
def login():
    form = SearchAS()
#    if form.validate_on_submit():
#        flash('Es impresionante la cantidad de AS que cumplen con lo requerido')
#        flash('From: Threshold={}, month={}, year={}'.format(form.from_threshold.data, form.from_month.data, form.from_year.data))
#        flash('to: Threshold={}, month={}, year={}'.format(form.to_threshold.data, form.to_month.data, form.to_year.data))
#        flash('Maximum months of grow={}'.format(form.maximum_months.data))    
#        df = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))
#        return render_template('tables.html',titles='Queries', tables=[df.to_html(classes='results')],form=form)
#    else:
    if form.validate_on_submit():
        validator=1
    else:
        validator=0
    if(is_float(form.from_threshold.data)):
        from_thr=float(form.from_threshold.data)
        if(from_thr > 1):
            flash('From Threshold must be less or equal than 1')
            validator=0
        if(from_thr  < 0):
            flash('From Threshold must be greater or equal than 0')
            validator=0
    else:
        flash('From Threshold must be float')
        validator=0
    if(is_float(form.to_threshold.data)):
        to_thr=float(form.to_threshold.data)
        if(to_thr > 1):
            flash('To Threshold must be less or equal than 1')
            validator=0
        if(to_thr  < 0):
            flash('To Threshold must be greater or equal than 0')
            validator=0
    else:
        flash('To Threshold must be float')
        validator=0
    if(validator==1):
        if(to_thr <= from_thr):
            flash('To Threshold must be greater than From Threshold')
            validator=0
    from_date=int(form.from_year.data)*12+int(form.from_month.data)
    to_date=int(form.to_year.data)*12+int(form.to_month.data)
    if not (to_date > from_date):
        flash('To Date must be greater than From Date')
        validator=0
    max_months=to_date - from_date
    if (is_int(form.maximum_months.data)):
        if (max_months < form.maximum_months.data):
            flash('To Date - From Date > Maximum Months')
            validator=0
    else:
        flash('Maximum Months must be an int')
    if (validator==1):
        flash('Es impresionante la cantidad de AS que cumplen con lo requerido, se muestran en esta preciosa tabla:')
        flash('From: Threshold={}, month={}, year={}'.format(form.from_threshold.data, form.from_month.data, form.from_year.data))
        flash('to: Threshold={}, month={}, year={}'.format(form.to_threshold.data, form.to_month.data, form.to_year.data))
        flash('Maximum months of grow={}'.format(form.maximum_months.data))
        columns='[{title:"AS Number", field:"ASNumber"},{title:"Short Name", field:"ShortName"},{title:"AS Type", field:"type"},{title:"Country", field:"Country", align:"center"},{title:"From Th.='+form.from_threshold.data+'",field:"StartGrow"},{title:"To Th.='+form.to_threshold.data+'",field:"StopGrow"},{title:"Grow Months",field:"MonthGrow"},],'
        return render_template('tables.html',form=form, columns=columns)
        #return render_template('form.html', title='Sign In', form=form)
    else:
        return render_template('form.html', title='Sign In', form=form)

@app.route('/query', methods=['GET'])
def query():
    #data=df.head() 
    from_month=request.args["from_month"] 
    from_year=request.args["from_year"]
    from_th=request.args["from_th"]
    to_month=request.args["to_month"]
    to_year=request.args["to_year"]
    to_th=request.args["to_th"]
    maximum=request.args["maximum"]
    data=select_as(asn_df, core_df, int(from_month)+(int(from_year)-1998)*12, float(from_th), int(to_month)+(int(to_year)-1998)*12,float(to_th), int(maximum)) 
    js = data.to_json(orient='records')
    resp = Response(js, status=200, mimetype='application/json')
    resp.headers['Link'] = 'http://itba.edu.ar'
    return resp

@app.route('/plot/as_evolution.png', methods=['GET'])
def plot():
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    x = [dt.date(1998,1,1)+relativedelta(months=int(round(i))-1) for i in core_df.index.tolist()]  
    y = core_df["15169"].tolist() 
    y2= core_df["2"].tolist() 
    fig, ax = plt.subplots(figsize=(20, 10))
    ASs=request.args.get('ASs')
    from_th=float(request.args.get('from_th')) 
    to_th=float(request.args.get('to_th'))
    if (ASs != None):
        ASs=ASs.split(",")
        for asn in ASs:
            ax.plot(x, core_df[asn].tolist(),label="ASN "+asn)
        ax.plot([x[0],x[-1]],[from_th,from_th],"k-", linewidth=2)
        ax.plot([x[0],x[-1]],[to_th,to_th],"k-", linewidth=2) 
   # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    datemin = dt.date(1998, 1, 1)
    datemax = dt.date(2017, 10, 1)
    ax.set_xlim(datemin, datemax)
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    #ax.format_ydata = price
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    fig.autofmt_xdate()
    plt.title("K-Core evolution for selected ASs")
    plt.ylabel("Normalized K-Core")
    plt.xlabel("Time")
    png_output=BytesIO()
    fig.savefig(png_output,format='png')
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response
