import plotly.express as px
import plotly.graph_objects as go
import  plotly as py
import pandas as pd
import sys
from datetime import datetime
from datetime import timedelta 

def subtract_dates(date1, date2="2020-03-01", date_format="%Y-%m-%d"):
  """
  Takes two dates %m/%d/%Y format. Returns date1 - date2, measured in days.
  """
  a = datetime.strptime(date1, date_format)
  b = datetime.strptime(date2, date_format)
  delta = a - b
  return delta.days


df = pd.read_csv(sys.argv[1], delimiter=',')
# fig = px.line(df, x="#time", y="susceptible", title='COVID-19 Simulation - London Borough of Brent')
# fig = px.line(df, x="#time", y="exposed", title='COVID-19 Simulation - London Borough of Brent')
# py.offline.plot(fig, filename='name.html')

#df['new cases'] = df['exposed'].diff(1) + df['infectious'].diff(1)
validation = pd.read_csv(sys.argv[2], delimiter=',')
for d in validation['Date']:
  print(d)
  print(subtract_dates(d))
#print(simdays)
sys.exit()

layout = go.Layout(yaxis=(dict(type='log',autorange=True)))

fig = go.Figure(layout=layout)

# Add traces
fig.add_trace(go.Scatter(x=df['#time'], y=df['num infections today'],
                    mode='lines+markers',
                    name='# of new infections (sim)',  line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df['#time'], y=df['num hospitalisations today'],
                    mode='lines+markers',
                    name='# of new hospitalisations (sim)',  line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['#time'], y=df['num hospitalisations today (data)'],
                    mode='lines+markers',
                    name='# of new hospitalisations (data)',  line=dict(color='dark blue')))
#fig.add_trace(go.Bar(x=df['#time'], y=df['num hospitalisations today (data)'],
#                    name='# of new hospitalisations (data)'))

py.offline.plot(fig, filename='{}-DailyChart.html'.format(sys.argv[2]))
