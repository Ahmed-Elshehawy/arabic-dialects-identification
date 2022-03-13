# import required libraries
import pandas as pd
import dash
import torch
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_daq as daq
import app_helpers as ah
import nlp_helpers as nh
import warnings
import base64
import io
warnings.filterwarnings("ignore")


vocab_to_int_no_stem = nh.load_pickle_file('no_stem/vocab_to_int_no_stem.obj')
int_to_vocab_no_stem = nh.load_pickle_file('no_stem/int_to_vocab_no_stem.obj')
int_to_labels = nh.load_pickle_file('int_to_labels.obj')

############################### Load lstm model no stem ############################
use_cuda = torch.cuda.is_available()
weighths_path = 'no_stem/models/model.pt'
vocab_size = len(vocab_to_int_no_stem) + 1
output_size = len(int_to_labels)
embedding_dim = 400
hidden_dim = 256
n_layers = 2
seq_length = 20
drop_prob = 0.5

model_lstm_no_stem = ah.load_lstm_model(weighths_path, vocab_size, output_size, embedding_dim,
                                        hidden_dim, n_layers, seq_length, drop_prob, use_cuda)
                                        
#model_ml_no_stem = ah.load_ml_model()                                        


app = dash.Dash(name='Arabic_Dialect_Classification')

app.layout = html.Div(children=[
    # title
    html.H1(
        'Arabic Dialect Classification',
        style={'textAlign':'center', 'color': '#7161ef'}
    ), # title
    
    
    # First section (upload file, select model, and stemming) ##############################
    html.Div([
        # 1- upload file
        html.Div([dcc.Upload(id='upload_data', children=html.Div([
            'Drag and Drop or click to ',
            html.A('Select a File')
        ]), style={
                'width': '100%',
                'height': '44px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                
            })
        ], style={'margin': 'auto', 'width': '50%'}), # 1- Upload file
        
        
        # 2- select model
        html.Div([dcc.Dropdown(
            id='model-type',
            options=[
                {'label': 'LSTM Model', 'value': 'lstm'}, 
                {'label': 'ML Model', 'value': 'ml'}
                ],
            placeholder='Select Model',
            style={'width': '100%', 'fontsize': '20px', 
                'padding': '5px', 'text-align-last': 'center'}
            )
        ], style={'margin': 'auto', 'width': '50%'}), # 2- select file        
    ], style={'display': 'flex', 'margin': 'auto'}), # end of first section
    ################################################################################
    
    # second section (graph and download button) ###################################
    html.Div([
        # 1- the graph
        html.Div([dcc.Graph()], id='plot1', 
            style={'margin': 'auto','width': '98%'}), # 1- the graph
            
        #2- Download button
        html.Div([
            html.Button('Download Result', id='dwn_btn'),
            dcc.Download(id='dwn-result')      
        ])
    ], style={'display': 'flex', 'margin': 'auto'}), # end of second section
    ####################################################################3######
        
    # third section ###############################################
    html.Div([
        # title
        html.H3('Test the model: '),
        # text box
        html.Div([
            dcc.Textarea(id='input-text',
            placeholder='Enter you text', style={'width': '50%', 'height': 150}),
            
            # output
            html.Div([
            html.H2(id='output')
            ], style={'margin': 'auto', 'color': '#7161ef', 'textAlign':'center'})
            
            
        ], style={'display': 'flex'})
        
        #html.Div([dcc.Graph(figure=plot1, )], id='plot1', 
        #style={'margin': 'auto','width': '50%'})      

    ])

], style={'backgroundColor':'#FFFFFF'})




#############################################
def parse_content(content, filename):
    df = pd.DataFrame()
    content_type, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
    return df                
#####################################
old_content = ''
old_model_type = ''
fig_count = px.bar()
result_df = pd.DataFrame()
old_n_clicks = 0
@app.callback(
                [Output(component_id='output', component_property='children'),
                Output(component_id='plot1', component_property='children'),
                Output(component_id='dwn-result', component_property='data')],
                
                [Input(component_id='upload_data', component_property='contents'),
                Input(component_id='input-text', component_property='value'),
                Input(component_id='model-type', component_property='value'),
                Input(component_id='dwn_btn', component_property='n_clicks')],
                
                [State('upload_data', 'filename'),
                State('upload_data', 'last_modified')]
)
def update_output(content, inp_text, model_type, n_clicks, name, date):
    global old_content
    global old_model_type
    global fig_count
    global old_n_clicks
    global result_df 
    
    res = None
    if model_type is not None:
        # this condition will triger if:
        #   1- you have uploaded a file for the first time, and selected a model type
        #   2- you changed the model type or upoloaded a new file
        if (content is not None) and (model_type != old_model_type or content != old_content):
            old_model_type = model_type
            old_content = content

            df_uploaded = parse_content(content, name)
            
            data = ah.preprocess_data_no_stem(df_uploaded.head(500), 3)
            if model_type == 'lstm':
                features_tensor = ah.prepare_data_lstm(data, vocab_to_int_no_stem, seq_length=seq_length)
                pred = ah.predict_lstm(model_lstm_no_stem, features_tensor, use_cuda)
                pred = pd.Series(pred).map(int_to_labels)
            
            elif model_type == 'ml':
                features = ah.prepare_data_ml(data)
                pred = ah.predict_ml(model_ml_no_stem, features)


            data['pred_dialect'] = pred
            
            grouped = data.groupby('pred_dialect').count()
            fig_count = px.bar(grouped, x=grouped.index, y='text', title='Dialect Count', 
                               labels={'text': 'Count', 'pred_dialect': 'Predicted Dialect'},
                               category_orders={'pred_dialect': data['pred_dialect'].value_counts().index})
                               

            result_df['text'] = df_uploaded['text']
            result_df['pred_dialect'] = data['pred_dialect']

            
        #if inp_text != old_text and isinstance(inp_text,str):
        
        if isinstance(inp_text,str):
            data_box = pd.DataFrame()
            data_box['text'] = [inp_text]
            data_box = ah.preprocess_data_no_stem(data_box, 3)
            if model_type == 'lstm':                              
                features_tensor = ah.prepare_data_lstm(data_box, vocab_to_int_no_stem, seq_length=20)
                pred = ah.predict_lstm(model_lstm_no_stem, features_tensor, use_cuda).item()
                pred = int_to_labels[pred]

            elif model_type == 'ml':
                                           
                features = ah.prepare_data_ml(data_box)
                pred = ah.predict_ml(model_ml_no_stem).item()

            res = 'Your Dialect is: \n{}'.format(pred)
    
    
    if (n_clicks is not None) and (n_clicks > old_n_clicks):
        old_n_clicks = n_clicks  
            
        return res, dcc.Graph(figure=fig_count), dcc.send_data_frame(result_df.to_csv, 'result.csv')    

    return res, dcc.Graph(figure=fig_count), None

if __name__ == '__main__':
    print('Serving on: http://localhost:1111/')
    #app.run_server(host='172.17.0.2', port=1111, debug=False, dev_tools_ui=False, dev_tools_props_check=False)
    app.run_server(port=1111, debug=False, dev_tools_ui=False, dev_tools_props_check=False)

