import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from umap import UMAP
import numpy as np
import os
import sys

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

        self.reducer = UMAP(n_components=2, n_neighbors=8, min_dist=0.1, spread=0.12)
        
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
                ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-right': '20px'}),

                # Plot
                html.Div([
                    dcc.Graph(
                        id='scatter-plot',
                        figure={'data': [self.default_trace], 'layout': self.layout},
                        config={'displayModeBar': False}  # Turn off the toolbar
                    )
                ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
            ], style={'display': 'flex', 'padding-top': '10px'}),

            dcc.Store(id='relevant-chunks-index-map', data={}), 
            dcc.Store(id='non-relevant-chunks-index-map', data={i: i for i in range(len(self.chunks))}), 
        ])

    def setup_callbacks(self):
        # Define callback to update the plot with the query embedding, model answer, and augmented prompt
        @self.app.callback(
            [Output('scatter-plot', 'figure'), Output('model-answer', 'children'), Output('query-input', 'value'),
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
                
                return figure, model_answer, "", relevant_chunks_index_map, non_relevant_chunks_index_map

            # Return the original figure, empty model answer, and empty augmented prompt if no query is submitted
            return [dash.no_update for _ in range(5)]

    def run(self):
        # Run the Dash app
        self.app.run_server(debug=True)
