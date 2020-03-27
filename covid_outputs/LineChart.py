import plotly.express as px
import plotly.graph_objects as go
import  plotly as py
import pandas as pd


df = pd.read_csv('covid_out.csv', delimiter=',')
# fig = px.line(df, x="#time", y="susceptible", title='COVID-19 Simulation - London Borough of Brent')
# fig = px.line(df, x="#time", y="exposed", title='COVID-19 Simulation - London Borough of Brent')
# py.offline.plot(fig, filename='name.html')


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=df['#time'], y=df['susceptible'],
                    mode='lines+markers',
                    name='susceptible',  line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df['#time'], y=df['exposed'],
                    mode='lines+markers',
                    name='exposed',  line=dict(color='purple')))
fig.add_trace(go.Scatter(x=df['#time'], y=df['infectious'],
                    mode='lines+markers',
                    name='infectious',  line=dict(color='red')))
fig.add_trace(go.Scatter(x=df['#time'], y=df['recovered'],
                    mode='lines+markers',
                    name='recovered', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df['#time'], y=df['dead'],
                    mode='lines+markers',
                    name='dead', line=dict(color='black')))

py.offline.plot(fig, filename='name.html')