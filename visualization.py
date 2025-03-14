import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from umap import UMAP
import numpy as np

class TestClass:
    def __init__(self):
        pass

class VisualizationApp:
    def __init__(self, chunks, embedding_model, prompt_model_function, initial_query=""):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.prompt_model_function = prompt_model_function
        self.query = initial_query

        self.embeddings = embedding_model.embed_documents(chunks)

        self.reducer = UMAP(n_components=2, n_neighbors=8, min_dist=0.1, spread=0.12, random_state=42)
        self.pca_embeddings = self.reducer.fit_transform(self.embeddings)

        # Update the layout
        self.layout = go.Layout(
            hovermode='closest',
            showlegend=True,
            autosize=True,
            margin=dict(l=20, r=0, b=20, t=0),  # Reduce the top margin to remove blank space
            legend=dict(
                x=0,  # Position the legend at the bottom left corner
                y=0,
                xanchor='left',
                yanchor='bottom',
                orientation='v'  # Horizontal orientation
            )
        )

        # Define default trace for when no query is submitted
        self.default_trace = go.Scatter(
            x=self.pca_embeddings[[i for i in range(len(chunks))], 0],
            y=self.pca_embeddings[[i for i in range(len(chunks))], 1],
            customdata=["non-relevant"] * len(chunks),
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.7),
            text=chunks,
            hoverinfo='text', 
            textposition='top center',
            name='Info-Chunks'
        )

        # Create the Dash app
        self.app = dash.Dash(__name__)

        self.setup_layout()
        self.setup_callbacks()


    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                # Text Input for query and Model Answer
                html.Div([
                    html.H4("Frage:", style={'margin-top': '5px', 'margin-bottom': '5px', 'font-size': '14px'}),
                    
                    dcc.Loading(
                        id="loading-spinner",
                        type="circle", 
                        children=[
                            html.Div([ 
                                dcc.Input(id='query-input', value=self.query, placeholder='Hier Frage eingeben', style={'width': '80%', 'border': '1px solid lightgrey', 'height': '50px', 'display': 'inline-block'}),
                                html.Button('Submit', id='submit-button', n_clicks=0, style={'width': '20%', 'height': '54px', 'display': 'inline-block'}), 
                        ], style={'display': 'flex'})]
                    ),
                    html.H4("Antwort des Modells:", style={'margin-top': '20px', 'margin-bottom': '5px', 'font-size': '14px'}),
                    html.Div(id='model-answer', style={'white-space': 'pre-line', 'border': 'none', 'font-size': '14px', 'height': '250px', 'padding': '10px', 'border': 'none', 'overflowY': 'scroll'})
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-right': '20px'}),

                # Plot
                html.Div([
                    dcc.Graph(
                        id='scatter-plot',
                        figure={'data': [self.default_trace], 'layout': self.layout},
                        config={'displayModeBar': False}  # Turn off the toolbar
                    )
                ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.Div("Deine Mudda In dieser Grafik sind die Informations-Chunks aus dem 3072-dimensionalen Embedding-Space entlang der 2 aussagekräftigsten Dimensionen dargestellt (PCA). Hover über einem Punkt im Plot um die zugehörige Information anzuzeigen.", style={'font-size': '11px', 'color': 'gray'}),

                    html.Div([
                    html.P("Verwendete Informations-Chunks:", style={'margin-top': '15px', 'margin-bottom': '5px', 'font-size': '14px', 'color': 'black'}),

                    html.Div(id='chunk-info-1', style={'white-space': 'pre-line', 'font-size': '12px', 'color': '#e6840e', 'height': '40px', 'overflowY': 'scroll', 'margin-bottom': '5px', 'border': '1px solid lightgrey'}), 
                    html.Div(id='chunk-info-2', style={'white-space': 'pre-line', 'font-size': '12px', 'color': '#e6840e', 'height': '40px', 'overflowY': 'scroll', 'margin-bottom': '5px', 'border': '1px solid lightgrey'}), 
                    html.Div(id='chunk-info-3', style={'white-space': 'pre-line', 'font-size': '12px', 'color': '#e6840e', 'height': '40px', 'overflowY': 'scroll', 'margin-bottom': '5px', 'border': '1px solid lightgrey'}), 
                    html.Div(id='chunk-info-4', style={'white-space': 'pre-line', 'font-size': '12px', 'color': '#e6840e', 'height': '40px', 'overflowY': 'scroll', 'margin-bottom': '5px', 'border': '1px solid lightgrey'}), 
                    html.Div(id='chunk-info-5', style={'white-space': 'pre-line', 'font-size': '12px', 'color': '#e6840e', 'height': '40px', 'overflowY': 'scroll', 'margin-bottom': '5px', 'border': '1px solid lightgrey'}), 
                    ], id='used-context-chunks', style={'display': 'none'}),  # Initially hidden
            
                    html.Div([
                    html.P("Ausgewählter Informations-Chunk:", style={'margin-top': '15px', 'margin-bottom': '5px', 'font-size': '14px', 'color': 'black'}),
                    html.Div(id='chunk-text', style={'white-space': 'pre-line', 'font-size': '12px', 'color': 'darkblue'}), 
                    ]),  
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-left': '20px'})
                ], style={'display': 'flex', 'padding-top': '10px'}),


            html.Div([
                html.H4("Kompletter Augmented Prompt:", style={'margin-top': '5px', 'margin-bottom': '5px', 'font-size': '14px'}),
                html.Div(id='augmented-prompt'),
            ], style={'white-space': 'pre-line', 'border': 'none', 'padding': '10px', 'font-size': '12px', 'width': '60%', 'margin': 'auto', 'padding-top': '50px'}), 

            dcc.Store(id='relevant-chunks-index-map', data={}), 
            dcc.Store(id='non-relevant-chunks-index-map', data={i: i for i in range(len(self.chunks))}), 
        ])

    def setup_callbacks(self):
        # Define callback to update the text box when a point is clicked
        @self.app.callback(
            [Output('chunk-text', 'children'), Output('chunk-text', 'style')], 
            [Input('scatter-plot', 'hoverData')], 
            [State('relevant-chunks-index-map', 'data'), State('non-relevant-chunks-index-map', 'data')]
        )
        def display_chunk_text(hoverData, relevant_chunks_index_map, non_relevant_chunks_index_map):
            if hoverData is None:
                return "Hover über einem Punkt um die Chunk-Information anzuzeigen.", {'white-space': 'pre-line', 'font-size': '12px', 'color': 'black'}
            
            point_type = hoverData['points'][0]['customdata']

            if point_type == 'query':
                return hoverData['points'][0]['text'], {'white-space': 'pre-line', 'font-size': '12px', 'color': 'red'}

            # Extract the index of the hovered point
            point_index = hoverData['points'][0]['pointIndex']
            point_index = str(point_index)

            # Check if the point is in the relevant or non-relevant chunks
            if point_type == 'relevant':
                return self.chunks[relevant_chunks_index_map[point_index]], {'white-space': 'pre-line', 'font-size': '12px', 'color': '#e6840e'}
            elif point_type == 'non-relevant':
                return self.chunks[non_relevant_chunks_index_map[point_index]], {'white-space': 'pre-line', 'font-size': '12px', 'color': 'darkblue'}

            return None

        # Define callback to update the plot with the query embedding, model answer, and augmented prompt
        @self.app.callback(
            [Output('scatter-plot', 'figure'), Output('model-answer', 'children'), Output('augmented-prompt', 'children'), Output('query-input', 'value'), Output('used-context-chunks', 'style'),
            Output('chunk-info-1', 'children'), Output('chunk-info-2', 'children'), Output('chunk-info-3', 'children'), Output('chunk-info-4', 'children'), Output('chunk-info-5', 'children'), 
            Output('relevant-chunks-index-map', 'data'), Output('non-relevant-chunks-index-map', 'data')],
            [Input('submit-button', 'n_clicks'), Input('query-input', 'n_submit')],
            [State('query-input', 'value')], 
            prevent_initial_call=True
        )
        def update_plot(n_clicks, n_submit, query):
            do_something = False

            if n_submit and query:
                do_something = True

            if n_clicks > 0 and query:
                do_something = True

            if do_something:
                # Get the model answer, context indices, and augmented prompt
                model_answer, context_indices, augmented_prompt = self.prompt_model_function(query)
                
                # Create a new trace for the query embedding
                query_embedding = self.embedding_model.embed_query(query)
                query_pca = self.reducer.transform(np.array([query_embedding]))[0]
                query_trace = go.Scatter(
                    x=[query_pca[0]],
                    y=[query_pca[1]],
                    customdata=['query'], 
                    mode='markers',
                    marker=dict(size=10, color='red', opacity=0.7),
                    text=[query],  # Add the query text to display on hover
                    hoverinfo='text', 
                    textposition='top center',
                    name='Frage'
                )
                
                # Separate the chunks into relevant and non-relevant
                relevant_chunks = [i for i in range(len(self.chunks)) if i in context_indices]
                relevant_chunks_index_map = {i: ind for i, ind in enumerate(relevant_chunks)}

                non_relevant_chunks = [i for i in range(len(self.chunks)) if i not in context_indices]
                non_relevant_chunks_index_map = {i: ind for i, ind in enumerate(non_relevant_chunks)}

                chunk_infos = [self.chunks[i] for i in relevant_chunks]
                
                # Create traces for relevant and non-relevant chunks
                relevant_trace = go.Scatter(
                    x=self.pca_embeddings[relevant_chunks, 0],
                    y=self.pca_embeddings[relevant_chunks, 1],
                    customdata=["relevant"] * len(relevant_chunks),
                    mode='markers',
                    marker=dict(size=10, color='#e6840e', opacity=0.7),
                    text=[self.chunks[i] for i in relevant_chunks],
                    hoverinfo='text', 
                    textposition='top center',
                    name='Relevante Chunks'
                )
                
                non_relevant_trace = go.Scatter(
                    x=self.pca_embeddings[non_relevant_chunks, 0],
                    y=self.pca_embeddings[non_relevant_chunks, 1],
                    customdata=["non-relevant"] * len(non_relevant_chunks),
                    mode='markers',
                    marker=dict(size=10, color='blue', opacity=0.7),
                    text=[self.chunks[i] for i in non_relevant_chunks],
                    hoverinfo='text',
                    textposition='top center',
                    name='Info-Chunks'
                )
                
                # Update the figure with the new traces
                figure = {
                    'data': [non_relevant_trace, relevant_trace, query_trace],
                    # 'data': [non_relevant_trace, relevant_trace],
                    'layout': self.layout
                }
                
                return figure, model_answer, augmented_prompt, "", {'display': 'block'}, *chunk_infos, relevant_chunks_index_map, non_relevant_chunks_index_map

            # Return the original figure, empty model answer, and empty augmented prompt if no query is submitted
            return [dash.no_update for _ in range(12)]

    def run(self):
        # Run the Dash app
        self.app.run_server(debug=True)
