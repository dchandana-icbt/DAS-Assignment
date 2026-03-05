import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import networkx as nx
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

try:
    from scipy.spatial import KDTree
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


st.set_page_config(
    page_title='A/L Student Results Dashboard',
    page_icon='🎓',
    layout='wide',
)

SUBJECT_SELECTION_COLS = [
    'Combined_Mathematics', 'Physics', 'Chemistry', 'Biology', 'Agricultural_Science',
    'Accounting', 'Business_Studies', 'Business_Statistics', 'Economics',
    'Engineering_Technology', 'Bio_Systems_Technology', 'Science_for_Technology',
    'ICT', 'Sinhala', 'Political_Science', 'History', 'Geography',
    'Logic_Scientific_Method', 'English_Literature', 'Buddhist_Civilization', 'Media_Studies'
]

OL_GRADE_COLS = [
    'results_OL_Buddhism', 'results_OL_Christianity', 'results_OL_Islam',
    'results_OL_Sinhala', 'results_OL_Tamil', 'results_OL_English',
    'results_OL_Mathematics', 'results_OL_History', 'results_OL_Science',
    'results_OL_BusinessAccountingStudies', 'results_OL_Geography',
    'results_OL_Music', 'results_OL_Art', 'results_OL_Dancing',
    'results_OL_ICT', 'results_OL_AFT'
]

NUMERIC_CANDIDATES = [
    'actual_School_Attendency_AL_Classes(%)',
    'grade1_Avg_Marks', 'grade2_Avg_Marks', 'grade3_Avg_Marks', 'grade4_Avg_Marks',
    'grade5_Avg_Marks', 'grade6_Avg_Marks', 'grade7_Avg_Marks', 'grade8_Avg_Marks',
    'grade9_Avg_Marks', 'grade10_Avg_Marks', 'grade11_Avg_Marks',
    'grade12_Subject1_Paper1_Marks', 'grade12_Subject1_Paper2_Marks',
    'grade12_Subject2_Paper1_Marks', 'grade12_Subject2_Paper2_Marks',
    'grade12_Subject3_Paper1_Marks', 'grade12_Subject3_Paper2_Marks',
    'weekly_Library_Hours', 'weekly_Study_Time', 'attendance_AL_Classes(%)',
    'current_Stress_Level(1-5)', 'AL_Exam_Year', 'age_at_exam'
]

GRADE_MAP = {'A': 5, 'B': 4, 'C': 3, 'S': 2, 'W': 1}


def hash_to_coord(text, center=(7.486, 80.364), radius=0.18):
    h = hashlib.md5(text.encode('utf-8')).hexdigest()
    a = int(h[:8], 16) / 16**8
    r = int(h[8:16], 16) / 16**8
    angle = 2 * math.pi * a
    dist = radius * (0.15 + 0.85 * r)
    lat = center[0] + dist * math.cos(angle)
    lon = center[1] + dist * math.sin(angle)
    return lat, lon


@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(Path('data/student_data_with_final_status.csv'))
    except Exception as e:
        st.error(f'Data loading error: {e}')
        st.stop()

    df = df.copy()

    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'date_Of_Birth' in df.columns:
        df['date_Of_Birth'] = pd.to_datetime(df['date_Of_Birth'], errors='coerce', dayfirst=True)

    if 'Final_Status' in df.columns:
        df['Final_Status'] = df['Final_Status'].astype(str).str.strip().str.title()
    else:
        df['Final_Status'] = 'Unknown'

    if 'Stream' in df.columns:
        df['Stream'] = df['Stream'].astype(str).str.strip()
    else:
        df['Stream'] = 'Unknown'

    if 'sex' not in df.columns:
        df['sex'] = 'Unknown'
    if 'address' not in df.columns:
        df['address'] = 'Unknown'
    if 'id' not in df.columns:
        df['id'] = [f'S{i+1}' for i in range(len(df))]
    if 'name' not in df.columns:
        df['name'] = [f'Student {i+1}' for i in range(len(df))]

    for col in SUBJECT_SELECTION_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col + '_selected'] = df[col].str.lower().eq('selected')
        else:
            df[col + '_selected'] = False

    for col in OL_GRADE_COLS:
        if col in df.columns:
            df[col + '_score'] = df[col].astype(str).str.strip().str.upper().map(GRADE_MAP)

    score_parts = [c for c in [
        'grade10_Avg_Marks', 'grade11_Avg_Marks',
        'grade12_Subject1_Paper1_Marks', 'grade12_Subject1_Paper2_Marks',
        'grade12_Subject2_Paper1_Marks', 'grade12_Subject2_Paper2_Marks',
        'grade12_Subject3_Paper1_Marks', 'grade12_Subject3_Paper2_Marks',
        'attendance_AL_Classes(%)', 'actual_School_Attendency_AL_Classes(%)'
    ] if c in df.columns]
    if score_parts:
        z = df[score_parts].apply(lambda s: (s - s.mean()) / (s.std() if pd.notna(s.std()) and s.std() != 0 else 1))
        df['Preparedness_Index'] = z.mean(axis=1)
    else:
        df['Preparedness_Index'] = np.nan

    sel_cols = [c + '_selected' for c in SUBJECT_SELECTION_COLS]
    df['Selected_Subject_Count'] = df[sel_cols].sum(axis=1)

    df[['lat', 'lon']] = df['address'].fillna('Unknown').apply(lambda x: pd.Series(hash_to_coord(str(x))))
    return df


def filter_df(df):
    st.sidebar.header('Filters')

    dff = df.copy()
    for label, col in [('A/L Exam Year', 'AL_Exam_Year'), ('Stream', 'Stream'), ('Final Status', 'Final_Status'), ('Sex', 'sex')]:
        if col in dff.columns:
            vals = sorted(dff[col].dropna().astype(str).unique().tolist())
            selected = st.sidebar.multiselect(label, vals, default=vals)
            if selected:
                dff = dff[dff[col].astype(str).isin(selected)]

    st.sidebar.metric('Filtered students', f'{len(dff):,}')
    return dff


def add_section_header(title, caption=None):
    st.subheader(title)
    if caption:
        st.caption(caption)


def normal_visuals(df):
    add_section_header('Statistical and explanatory visuals',
                       'This section includes more than 12 figures, including designed visuals such as a slope chart and small multiples.')

    figs = []

    try:
        status_counts = df['Final_Status'].value_counts().reset_index()
        status_counts.columns = ['Final_Status', 'Count']
        figs.append(px.bar(status_counts, x='Final_Status', y='Count', color='Final_Status', title='1. Pass vs Fail distribution'))
    except Exception as e:
        st.warning(f'Figure 1 skipped: {e}')

    try:
        if 'AL_Exam_Year' in df.columns:
            year_status = df.groupby(['AL_Exam_Year', 'Final_Status']).size().reset_index(name='Count')
            figs.append(px.bar(year_status, x='AL_Exam_Year', y='Count', color='Final_Status', barmode='stack', title='2. Result distribution by exam year'))
    except Exception as e:
        st.warning(f'Figure 2 skipped: {e}')

    try:
        if 'Stream' in df.columns:
            stream_rate = df.groupby('Stream')['Final_Status'].apply(lambda s: (s == 'Pass').mean()).reset_index(name='Pass_Rate')
            fig = px.bar(stream_rate.sort_values('Pass_Rate', ascending=False), x='Stream', y='Pass_Rate', title='3. Pass rate by stream', text_auto='.1%')
            fig.update_yaxes(tickformat='.0%')
            figs.append(fig)
    except Exception as e:
        st.warning(f'Figure 3 skipped: {e}')

    try:
        if 'sex' in df.columns:
            sex_status = df.groupby(['sex', 'Final_Status']).size().reset_index(name='Count')
            figs.append(px.bar(sex_status, x='sex', y='Count', color='Final_Status', barmode='group', title='4. Pass/Fail comparison by sex'))
    except Exception as e:
        st.warning(f'Figure 4 skipped: {e}')

    try:
        if 'travel_Time' in df.columns:
            travel_rate = df.groupby('travel_Time')['Final_Status'].apply(lambda s: (s == 'Pass').mean()).reset_index(name='Pass_Rate')
            fig = px.line(travel_rate, x='travel_Time', y='Pass_Rate', markers=True, title='5. Pass rate by travel time')
            fig.update_yaxes(tickformat='.0%')
            figs.append(fig)
    except Exception as e:
        st.warning(f'Figure 5 skipped: {e}')

    try:
        if 'attendance_AL_Classes(%)' in df.columns:
            figs.append(px.box(df, x='Final_Status', y='attendance_AL_Classes(%)', color='Final_Status', title='6. Attendance distribution by final status'))
    except Exception as e:
        st.warning(f'Figure 6 skipped: {e}')

    try:
        if 'weekly_Study_Time' in df.columns:
            figs.append(px.violin(df, x='Final_Status', y='weekly_Study_Time', color='Final_Status', box=True, title='7. Weekly study time by final status'))
    except Exception as e:
        st.warning(f'Figure 7 skipped: {e}')

    try:
        if 'current_Stress_Level(1-5)' in df.columns:
            stress = df.groupby(['current_Stress_Level(1-5)', 'Final_Status']).size().reset_index(name='Count')
            figs.append(px.bar(stress, x='current_Stress_Level(1-5)', y='Count', color='Final_Status', barmode='group', title='8. Stress level distribution by result'))
    except Exception as e:
        st.warning(f'Figure 8 skipped: {e}')

    try:
        slope = df.groupby(['Stream', 'sex'])['Final_Status'].apply(lambda s: (s == 'Pass').mean()).reset_index(name='Pass_Rate')
        sexes = sorted(slope['sex'].dropna().unique().tolist())
        if len(sexes) >= 2:
            fig = go.Figure()
            for stream in slope['Stream'].unique().tolist():
                sub = slope[slope['Stream'] == stream].set_index('sex').reindex(sexes[:2])
                if sub['Pass_Rate'].notna().sum() == 2:
                    fig.add_trace(go.Scatter(x=sexes[:2], y=sub['Pass_Rate'], mode='lines+markers+text', name=stream, text=[stream, ''], textposition='middle left'))
            fig.update_layout(title='9. Designed visual — slope chart of pass rate by sex across streams', yaxis_tickformat='.0%', showlegend=False)
            figs.append(fig)
    except Exception as e:
        st.warning(f'Figure 9 skipped: {e}')

    try:
        if 'AL_Exam_Year' in df.columns and 'Stream' in df.columns:
            yearly_stream = df.groupby(['AL_Exam_Year', 'Stream'])['Final_Status'].apply(lambda s: (s == 'Pass').mean()).reset_index(name='Pass_Rate')
            fig = px.line(yearly_stream, x='AL_Exam_Year', y='Pass_Rate', facet_col='Stream', facet_col_wrap=3, markers=True, title='10. Designed visual — small multiples of pass rate by year and stream')
            fig.update_yaxes(tickformat='.0%')
            figs.append(fig)
    except Exception as e:
        st.warning(f'Figure 10 skipped: {e}')

    try:
        grade_cols = [f'grade{i}_Avg_Marks' for i in range(1, 12) if f'grade{i}_Avg_Marks' in df.columns]
        if grade_cols:
            g = df.groupby('Final_Status')[grade_cols].mean().T.reset_index()
            g.columns = ['Grade'] + g.columns[1:].tolist()
            fig = go.Figure()
            for col in g.columns[1:]:
                fig.add_trace(go.Scatter(x=g['Grade'], y=g[col], mode='lines+markers', name=col))
            fig.update_layout(title='11. Longitudinal average marks from Grade 1 to Grade 11')
            figs.append(fig)
    except Exception as e:
        st.warning(f'Figure 11 skipped: {e}')

    try:
        ol_score_cols = [c + '_score' for c in OL_GRADE_COLS if c + '_score' in df.columns]
        if ol_score_cols:
            ol_means = df.groupby('Final_Status')[ol_score_cols].mean().T
            ol_means.index = [i.replace('results_OL_', '').replace('_score', '') for i in ol_means.index]
            figs.append(px.imshow(ol_means, aspect='auto', title='12. Average O/L grade score heatmap by final status', color_continuous_scale='Blues'))
    except Exception as e:
        st.warning(f'Figure 12 skipped: {e}')

    try:
        if {'weekly_Study_Time', 'attendance_AL_Classes(%)', 'Final_Status'}.issubset(df.columns):
            figs.append(px.scatter(df, x='weekly_Study_Time', y='attendance_AL_Classes(%)', color='Final_Status', hover_data=[c for c in ['Stream', 'sex', 'address'] if c in df.columns], title='13. Study time vs attendance relationship'))
    except Exception as e:
        st.warning(f'Figure 13 skipped: {e}')

    try:
        corr_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
        if len(corr_cols) >= 4:
            corr = df[corr_cols].corr(numeric_only=True)
            figs.append(px.imshow(corr, aspect='auto', title='14. Correlation heatmap of numeric predictors', color_continuous_scale='RdBu', zmin=-1, zmax=1))
    except Exception as e:
        st.warning(f'Figure 14 skipped: {e}')

    try:
        miss = df.isna().mean().sort_values(ascending=False).head(20).reset_index()
        miss.columns = ['Column', 'Missing_Rate']
        fig = px.bar(miss, x='Missing_Rate', y='Column', orientation='h', title='15. Top 20 missingness rates')
        fig.update_xaxes(tickformat='.0%')
        figs.append(fig)
    except Exception as e:
        st.warning(f'Figure 15 skipped: {e}')

    try:
        if 'Stream' in df.columns:
            stream_stats = df.groupby('Stream').agg(n=('Final_Status', 'size'), pass_rate=('Final_Status', lambda s: (s == 'Pass').mean())).reset_index()
            stream_stats['se'] = np.sqrt((stream_stats['pass_rate'] * (1 - stream_stats['pass_rate'])) / stream_stats['n'].clip(lower=1))
            stream_stats['ci95'] = 1.96 * stream_stats['se']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=stream_stats['Stream'], y=stream_stats['pass_rate'], error_y=dict(type='data', array=stream_stats['ci95']), text=[f'n={n}' for n in stream_stats['n']]))
            fig.update_layout(title='16. Stream pass rate with approximate 95% confidence intervals')
            fig.update_yaxes(tickformat='.0%')
            figs.append(fig)
    except Exception as e:
        st.warning(f'Figure 16 skipped: {e}')

    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)


def spatial_aggregates(df):
    g = df.groupby('address').agg(
        n=('id', 'size'),
        pass_rate=('Final_Status', lambda s: (s == 'Pass').mean()),
        pass_count=('Final_Status', lambda s: (s == 'Pass').sum()),
        fail_count=('Final_Status', lambda s: (s == 'Fail').sum()),
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        preparedness=('Preparedness_Index', 'mean')
    ).reset_index()
    g['pass_per_100'] = g['pass_rate'] * 100
    g['se'] = np.sqrt((g['pass_rate'] * (1 - g['pass_rate'])) / g['n'].clip(lower=1))
    g['ci95'] = 1.96 * g['se']
    return g


def make_choropleth_geojson(agg, cell_size=0.05):
    if agg.empty:
        return {'type': 'FeatureCollection', 'features': []}

    features = []
    used = {}
    for _, row in agg.dropna(subset=['lat', 'lon']).iterrows():
        gx = round(row['lon'] / cell_size) * cell_size
        gy = round(row['lat'] / cell_size) * cell_size
        used.setdefault((gx, gy), []).append(row.to_dict())

    for (gx, gy), rows in used.items():
        n = sum(r['n'] for r in rows)
        pass_count = sum(r['pass_count'] for r in rows)
        pass_rate = pass_count / n if n else 0
        poly = [
            [gx - cell_size / 2, gy - cell_size / 2],
            [gx + cell_size / 2, gy - cell_size / 2],
            [gx + cell_size / 2, gy + cell_size / 2],
            [gx - cell_size / 2, gy + cell_size / 2],
            [gx - cell_size / 2, gy - cell_size / 2],
        ]
        features.append({
            'type': 'Feature',
            'properties': {
                'pass_rate': float(pass_rate),
                'n': int(n),
                'label': f'Grid cell | n={n} | pass rate={pass_rate:.1%}'
            },
            'geometry': {'type': 'Polygon', 'coordinates': [poly]}
        })
    return {'type': 'FeatureCollection', 'features': features}


def geographic_visuals(df):
    add_section_header('Geographic and spatial analysis',
                       'The dataset has locality labels but no real coordinates, so the maps below use deterministic schematic coordinates derived from the address field. They are suitable for spatial pattern storytelling, not ground-truth navigation.')

    agg = spatial_aggregates(df)

    try:
        st.markdown('**1. Interactive locality map (Folium)**')
        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=10, tiles='CartoDB positron')
        cluster = MarkerCluster().add_to(m)
        sample = df.sample(min(600, len(df)), random_state=42)
        for _, r in sample.iterrows():
            color = 'green' if r['Final_Status'] == 'Pass' else 'red'
            folium.CircleMarker(location=[r['lat'], r['lon']], radius=4, color=color, fill=True, fill_opacity=0.7, popup=f"{r['name']} | {r['Stream']} | {r['Final_Status']}").add_to(cluster)
        st_folium(m, use_container_width=True, height=500)
    except Exception as e:
        st.error(f'Interactive map failed: {e}')

    try:
        st.markdown('**2. Choropleth-style classified grid map**')
        geojson = make_choropleth_geojson(agg)
        if not geojson or 'features' not in geojson or len(geojson['features']) == 0:
            st.warning('Choropleth map could not be generated because no spatial grid features were available after filtering.')
        else:
            m2 = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=10, tiles='CartoDB positron')
            choropleth_df = pd.DataFrame([{'label': f['properties']['label'], 'pass_rate': f['properties']['pass_rate']} for f in geojson['features']])
            folium.Choropleth(
                geo_data=geojson,
                data=choropleth_df,
                columns=['label', 'pass_rate'],
                key_on='feature.properties.label',
                fill_color='YlGnBu',
                fill_opacity=0.75,
                line_opacity=0.25,
                legend_name='Pass rate (grid-cell classification)'
            ).add_to(m2)
            folium.GeoJson(geojson, tooltip=folium.GeoJsonTooltip(fields=['label'])).add_to(m2)
            st_folium(m2, use_container_width=True, height=500)
            st.caption('Legend explanation: darker cells indicate higher pass rates; each cell aggregates nearby schematic localities.')
    except Exception as e:
        st.error(f'Choropleth map rendering failed: {e}')

    try:
        st.markdown('**3. Hotspot / density heatmap**')
        m3 = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=10, tiles='CartoDB positron')
        HeatMap([[r['lat'], r['lon'], 1] for _, r in df.iterrows()], radius=20, blur=15, min_opacity=0.4).add_to(m3)
        st_folium(m3, use_container_width=True, height=500)
    except Exception as e:
        st.error(f'Density heatmap failed: {e}')

    try:
        st.markdown('**4. Spatial pattern analysis — nearest-neighbor spacing**')
        if SCIPY_OK:
            coords = agg[['lat', 'lon']].dropna().to_numpy()
            if len(coords) > 5:
                tree = KDTree(coords)
                dists, _ = tree.query(coords, k=2)
                nn = dists[:, 1]
                observed = nn.mean()
                area = (agg['lat'].max() - agg['lat'].min()) * (agg['lon'].max() - agg['lon'].min())
                density = len(coords) / max(area, 1e-6)
                expected_random = 0.5 / np.sqrt(max(density, 1e-6))
                ratio = observed / expected_random if expected_random else np.nan
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = px.histogram(x=nn, nbins=20, title='Nearest-neighbor distance distribution across localities')
                    fig.add_vline(x=observed, line_dash='dash', annotation_text=f'Observed mean={observed:.3f}')
                    fig.add_vline(x=expected_random, line_dash='dot', annotation_text=f'Random expectation={expected_random:.3f}')
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.metric('NN ratio', f'{ratio:.2f}')
                    if ratio < 1:
                        st.write('Interpretation: schematic localities appear **more clustered** than a random spatial pattern.')
                    elif ratio > 1:
                        st.write('Interpretation: schematic localities appear **more dispersed** than a random spatial pattern.')
                    else:
                        st.write('Interpretation: schematic localities are close to a random pattern.')
            else:
                st.info('Not enough localities available for nearest-neighbor analysis.')
        else:
            st.info('SciPy is not available, so nearest-neighbor analysis is skipped.')
    except Exception as e:
        st.error(f'Spatial pattern analysis failed: {e}')

    try:
        st.markdown('**5. Locality normalization / uncertainty map**')
        top = agg.sort_values('n', ascending=False).head(50)
        fig5 = px.scatter_map(top, lat='lat', lon='lon', size='n', color='pass_per_100', hover_name='address', hover_data={'n': True, 'pass_per_100':':.1f', 'ci95':':.3f'}, color_continuous_scale='Viridis', zoom=8, height=550, title='Passes per 100 students by locality (bubble size = sample size)')
        st.plotly_chart(fig5, use_container_width=True)
        st.caption('Uncertainty cue: hover to inspect the approximate 95% confidence half-width (`ci95`). Smaller localities have higher uncertainty.')
    except Exception as e:
        st.error(f'Normalization map failed: {e}')


def build_subject_graph(df):
    sel_cols = [c for c in SUBJECT_SELECTION_COLS if c + '_selected' in df.columns]
    G = nx.Graph()
    G.add_nodes_from(sel_cols)
    if not sel_cols:
        return G
    selected_matrix = pd.DataFrame({c: df[c + '_selected'].astype(int) for c in sel_cols})
    for i, a in enumerate(sel_cols):
        for b in sel_cols[i + 1:]:
            w = int((selected_matrix[a] & selected_matrix[b]).sum())
            if w > 0:
                G.add_edge(a, b, weight=w)
    return G


def graph_to_plotly(G, pos, title=''):
    edge_x, edge_y = [], []
    for u, v, _ in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#999'), hoverinfo='none')

    node_x, node_y, text, size = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        deg = G.degree(n, weight='weight')
        size.append(8 + deg / 40 if pd.notna(deg) else 8)
        text.append(f'{n}<br>Weighted degree: {deg:.0f}')

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
        textposition='top center', hovertext=text, hoverinfo='text',
        marker=dict(size=size, color=size, colorscale='Viridis', showscale=False, line=dict(width=1, color='white'))
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=40, b=0))
    return fig


def network_visuals(df):
    add_section_header('Network and graph visualisation',
                       'These figures use a subject co-selection network built from the subject columns marked as `Selected`.')

    try:
        G = build_subject_graph(df)
        if G.number_of_edges() == 0:
            st.warning('No subject selection network could be built from the current filtered data.')
            return

        pos = nx.spring_layout(G, seed=42, weight='weight')
        st.plotly_chart(graph_to_plotly(G, pos, title='1. Overall subject co-selection network'), use_container_width=True)

        hub_nodes = [n for n, _ in sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)[:8]]
        H = G.subgraph(hub_nodes).copy()
        pos2 = nx.spring_layout(H, seed=42, weight='weight')
        st.plotly_chart(graph_to_plotly(H, pos2, title='2. Top hubs subgraph'), use_container_width=True)

        comms = list(nx.community.greedy_modularity_communities(G, weight='weight'))
        community_map = {}
        for i, comm in enumerate(comms):
            for n in comm:
                community_map[n] = i
        node_x, node_y, node_c, node_text = [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x); node_y.append(y); node_c.append(community_map.get(n, 0)); node_text.append(n)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#bbb'), hoverinfo='none'))
        fig3.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=node_text,
    textposition='top center',
    marker=dict(
        size=14,
        color=node_c,
        colorscale='Viridis',
        line=dict(width=1, color='white'),
        showscale=False
    )
))
        fig3.update_layout(title='3. Community structure in the subject network', xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig3, use_container_width=True)

        if 'AL_Exam_Year' in df.columns:
            years = sorted(df['AL_Exam_Year'].dropna().unique().tolist())[:4]
            if years:
                subplots = make_subplots(rows=2, cols=2, subplot_titles=[str(y) for y in years] + [''] * max(0, 4 - len(years)))
                for idx, year in enumerate(years, start=1):
                    row = 1 if idx <= 2 else 2
                    col = idx if idx <= 2 else idx - 2
                    Gy = build_subject_graph(df[df['AL_Exam_Year'] == year])
                    if Gy.number_of_edges() == 0:
                        continue
                    posy = nx.spring_layout(Gy, seed=42, weight='weight')
                    ex, ey = [], []
                    for u, v in Gy.edges():
                        x0, y0 = posy[u]; x1, y1 = posy[v]
                        ex += [x0, x1, None]; ey += [y0, y1, None]
                    subplots.add_trace(go.Scatter(x=ex, y=ey, mode='lines', line=dict(width=1, color='#bbb'), showlegend=False, hoverinfo='none'), row=row, col=col)
                    nx_, ny_, txt_ = [], [], []
                    for n in Gy.nodes():
                        x, y = posy[n]
                        nx_.append(x); ny_.append(y); txt_.append(n)
                    subplots.add_trace(go.Scatter(x=nx_, y=ny_, mode='markers+text', text=txt_, textposition='top center', marker=dict(size=10, color='teal'), showlegend=False), row=row, col=col)
                subplots.update_layout(title='4. Temporal evolution of the subject network by exam year', height=800)
                subplots.update_xaxes(visible=False); subplots.update_yaxes(visible=False)
                st.plotly_chart(subplots, use_container_width=True)

        center = max(G.degree(weight='weight'), key=lambda x: x[1])[0]
        ego = nx.ego_graph(G, center, radius=1)
        pose = nx.spring_layout(ego, seed=42, weight='weight')
        st.plotly_chart(graph_to_plotly(ego, pose, title=f'5. Explanatory ego network around the strongest hub: {center}'), use_container_width=True)

        order = [n for n, _ in sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)]
        adj = nx.to_pandas_adjacency(G, nodelist=order, weight='weight')
        st.plotly_chart(px.imshow(adj, title='6. Adjacency heatmap of subject co-selection intensity', aspect='auto', color_continuous_scale='Blues'), use_container_width=True)
    except Exception as e:
        st.error(f'Network section failed: {e}')


def student_lookup(df):
    add_section_header('Student lookup and result explorer')
    query = st.text_input('Search by student ID or name')
    if query:
        hits = df[df['id'].astype(str).str.contains(query, case=False, na=False) | df['name'].astype(str).str.contains(query, case=False, na=False)]
    else:
        hits = df.head(20)

    show_cols = [c for c in ['id', 'name', 'Stream', 'sex', 'address', 'Final_Status', 'Preparedness_Index'] if c in hits.columns]
    st.dataframe(hits[show_cols].head(50), use_container_width=True)

    if len(hits):
        selected = st.selectbox('Choose a student for detail view', hits['id'].astype(str).tolist())
        row = df[df['id'].astype(str) == selected].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric('Final Status', row.get('Final_Status', 'NA'))
        c2.metric('Stream', row.get('Stream', 'NA'))
        prep = row.get('Preparedness_Index', np.nan)
        c3.metric('Preparedness Index', f'{prep:.2f}' if pd.notna(prep) else 'NA')

        detail_cols = ['name', 'school', 'sex', 'date_Of_Birth', 'address', 'travel_Time', 'religion', 'weekly_Study_Time', 'attendance_AL_Classes(%)', 'current_Stress_Level(1-5)']
        detail = {c: row[c] for c in detail_cols if c in row.index}
        st.json(detail)

        selected_subjects = [c for c in SUBJECT_SELECTION_COLS if c in row.index and str(row[c]).strip().lower() == 'selected']
        st.write('**Selected subjects:**', ', '.join(selected_subjects) if selected_subjects else 'None')

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download filtered dataset as CSV', data=csv, file_name='filtered_student_results.csv', mime='text/csv')


def main():
    st.title('🎓 A/L Student Results Prediction Dashboard')
    st.caption('Streamlit dashboard for exploring predicted Pass/Fail outcomes, statistical patterns, schematic geographic patterns, and subject-selection networks.')

    uploaded = st.file_uploader('Optional: upload a CSV with the same schema', type=['csv'])
    df = load_data(uploaded)
    dff = filter_df(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Students', f'{len(dff):,}')
    c2.metric('Pass rate', f"{(dff['Final_Status'].eq('Pass').mean()):.1%}" if 'Final_Status' in dff.columns else 'NA')
    c3.metric('Avg study time', f"{dff['weekly_Study_Time'].mean():.1f}" if 'weekly_Study_Time' in dff.columns else 'NA')
    c4.metric('Avg attendance', f"{dff['attendance_AL_Classes(%)'].mean():.1f}%" if 'attendance_AL_Classes(%)' in dff.columns else 'NA')

    tabs = st.tabs(['Overview', 'Statistical visuals', 'Geographic analysis', 'Network analysis', 'Student lookup'])

    with tabs[0]:
        st.markdown("""
        **How to read this dashboard**
        - The statistical section focuses on cohort structure, relationships, uncertainty, and explanatory stories.
        - The geographic section uses **schematic** locality coordinates because the source file does not contain real geocodes.
        - The network section builds a **subject co-selection graph** from columns marked as `Selected`.
        """)
        st.dataframe(dff.head(20), use_container_width=True)

    with tabs[1]:
        normal_visuals(dff)

    with tabs[2]:
        geographic_visuals(dff)

    with tabs[3]:
        network_visuals(dff)

    with tabs[4]:
        student_lookup(dff)


if __name__ == '__main__':
    main()
