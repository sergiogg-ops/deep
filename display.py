import pandas as pd
import streamlit as st
import altair as alt
from seaborn import color_palette
from sys import argv

def color_generator(categories, label='cluster', palette='crest'):
    """
    Generate a list of n colors.
    """
    palette = color_palette(palette, len(categories)).as_hex()
    return alt.Color(
            f'{label}:N',
            scale=alt.Scale(
              domain=categories,
              range=palette
            ),
            legend=alt.Legend(title="Cluster")
         )

def bar_leaderboard(df, metric, colors_cluster):
    """
    Generate a bar chart for the leaderboard.
    """
    top = 1.1 * df[metric].max()
    bottom = 0.9 * df[metric].min()

    # Create a conditional rule definition
    rule = alt.Chart(df[df['comment'] == 'Baseline']).mark_rule(strokeDash=[8, 8], color='red', size=2).encode(
        x=alt.X('name:N', sort=None, title='Participant'),
    )

    bar = (
        alt.Chart(df)
        .mark_bar(clip=True)
        .encode(
            x=alt.X('name:N', sort=None, title='Participant'),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score', scale=alt.Scale(domain=[bottom, top])),
            color=colors_cluster,
            tooltip=['name', 'cluster', f'{metric}'],
        )
        .properties(title=f'{metric.upper()} by Participant', width=700, height=400)
    )

    # Layer the bar chart and the rule
    chart = alt.layer(bar, rule).configure_axisX(labelAngle=-90)
    st.altair_chart(chart, use_container_width=True)
  
def scatter_chart(df,metric, colors_cluster):
    """
    Generate a scatter chart for the leaderboard.
    """
    top = 1.1 * df[metric].max()
    bottom = 0.9 * df[metric].min()
    right = 1.1 * df['time'].max()
    left = 0.9 * df['time'].min()
    chart = (
        alt.Chart(df)
          .mark_circle(size=80)
          .encode(
            x=alt.X('time:Q', title='Run time (s)', scale=alt.Scale(domain=[left, right])),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score', scale=alt.Scale(domain=[bottom, top])),
            color=colors_cluster,
            tooltip=['name','cluster',f'{metric}','time']
          )
          .properties(title=f"{metric.upper()} vs Computation Time", width=700, height=400)
    )
    st.altair_chart(chart, use_container_width=True)

def cluster_chart(df, metric, colors_cluster, mode):
    """
    Generate a cluster chart for the leaderboard.
    """
    conf = {}
    if mode == 'mt':
        conf = {'bleu': 'mean', 'ter': 'mean'}
    else:
        conf = {'wer': 'mean', 'bwer': 'mean'}
    conf['comment'] = lambda x: 'Baseline' if any(x == 'Baseline') else None
    cluster_df = df.groupby('cluster').agg(conf).reset_index()
    top = 1.1 * cluster_df[metric].max()
    bottom = 0.9 * cluster_df[metric].min()
    bar = (
      alt.Chart(cluster_df)
      .mark_bar(clip=True)
      .encode(
          x=alt.X('cluster:N', sort=None, title='Cluster'),
          y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score',scale=alt.Scale(domain=[bottom, top])),
          color=colors_cluster,
          tooltip=['cluster',f'{metric}']
      )
      .properties(title=f'Average {metric.upper()} by Cluster', width=700, height=400)
    )

    rule = alt.Chart(cluster_df[cluster_df['comment'] == 'Baseline']).mark_rule(strokeDash=[8, 8], color='red', size=2).encode(
        x=alt.X('cluster:N', sort=None, title='Cluster'),
    )
    chart = alt.layer(bar, rule).configure_axisX(labelAngle=0)
    st.altair_chart(chart, use_container_width=True)

def main():
    if len(argv) != 3:
      st.write("Usage: python display.py <filename.csv> <mode>")
    filename = argv[1]
    MODE = argv[2]
    if MODE not in ['mt','htr']:
      st.write("Invalid mode. Use 'mt' for machine translation or 'htr' for handwriting recognition.")
      return

    df = pd.read_csv(filename)
    colors_cluster = color_generator(df['cluster'].unique(), palette='viridis')
    df['datetime'] = pd.to_datetime(df['datetime'])

    if MODE == 'mt':
      st.title('ARCHER - Machine Translation Evaluation Results')
    else:
      st.title('ARCHER - Handwritten Text Recognition Evaluation Results')
    st.logo('images/archer.png', icon_image='images/archer-short.png', link='https://archer-challenge.eu/')

    #################
    # LEADER BOARD
    #################
    with st.sidebar:
      st.header('Leader board')
      leaderborad = df[['position','name', 'cluster']].copy()
      st.table(leaderborad.sort_values('position').set_index('position'))

    ##################
    # FILTERS
    ##################
    # Select main metric
    opt = ('BLEU', 'TER') if MODE == 'mt' else ('BWER', 'WER')
    metric = st.selectbox(
        'Select the metric to display',
        opt
    ).lower()

    # Range of clusters
    # num_clusters = max(df['cluster'])
    # best_cluster, worse_cluster = st.slider("Displayed clusters", 1, num_clusters, (1, max(df['cluster'])))
    # df = df[(df['cluster'] >= best_cluster) & (df['cluster'] <= worse_cluster)]

    #################################
    # CHARTS
    #################################
    bar_tab, scatter_tab, cluster_tab = st.tabs(['Leader board', 'Quality vs Time', 'Clusters average'])

    with bar_tab:
      bar_leaderboard(df, metric, colors_cluster)

    with scatter_tab:
      scatter_chart(df, metric, colors_cluster)

    with cluster_tab:
      cluster_chart(df, metric, colors_cluster, MODE)

    #################################
    # TABLES
    #################################
    with st.expander("Show data", expanded=False):
      st.write(df)
      st.download_button(
          label="Download data",
          data=df.to_csv(index=False).encode('utf-8'),
          file_name='archer_results.csv',
          mime='text/csv',
      )

if __name__ == '__main__':
    main()