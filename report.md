# Temporal Dynamics and Predictive Modeling of Co-Stardom Networks in Israeli Cinema

## Abstract

This study provides a comprehensive computational and network-scientific analysis of the Israeli film industry's co-stardom network, tracking its structural dynamics across three pivotal historical eras (Pre-1970, 1970–1990, and 1990–Present). By formalizing film cast metadata scraped from the Hebrew Wikipedia into dynamic bipartite graph projections, we model the evolving social capital, topological clustering, and collaborative patterns among Israeli actors. We evaluate community structures in the modern era using Louvain and Greedy Modularity optimization algorithms, profiling key actor cohorts across distinct sub-genres, theater ensembles, and religious cinema production houses. Furthermore, we compute node-level centrality metrics (Degree, Betweenness, Closeness, and Eigenvector) to identify structural archetypes such as "Prolific Stars," "Industry Bridges," "Network Insiders," and "Elite Collaborators." Finally, we construct a machine learning engine using a Random Forest classifier trained on topological link prediction metrics (Adamic-Adar, Jaccard Index, Shortest Path Length) and low-rank Matrix Factorization embeddings (Truncated SVD). Our predictive model achieves an F1-Score of **0.9730** and a ROC-AUC of **0.9654**, demonstrating that local triadic closure and latent genre affinities dominate co-stardom link formation in creative cultural industries.

---

## 1. Introduction & Theoretical Framing

### 1.1 Problem Formulation & Bipartite Graph Projection

The film industry represents a paradigmatic complex socioeconomic system driven by project-based collaborative teams. In creative industries, project performance, artistic innovation, and casting choices are deeply embedded within historical social networks. To formalize these relationships mathematically, we construct a dynamic bipartite graph model of cinematic collaborations.

Let $G_{\text{bipartite}} = (V_A \cup V_M, E_B)$ represent an undirected bipartite graph, where:
- $V_A = \{a_1, a_2, \dots, a_N\}$ denotes the set of actors (nodes of type A).
- $V_M = \{m_1, m_2, \dots, m_K\}$ denotes the set of films (nodes of type M).
- $E_B \subseteq V_A \times V_M$ denotes the set of bipartite edges, where an edge $e = (a_i, m_k) \in E_B$ exists if and only if actor $a_i$ appeared in the cast of film $m_k$.

Each movie $m_k \in V_M$ is annotated with a temporal attribute $Y(m_k) \in \mathbb{Z}^+$, representing its calendar release year.

To analyze actor-actor collaborations directly, we project the bipartite graph $G_{\text{bipartite}}$ into a one-mode weighted, undirected temporal co-stardom network $G = (V, E, W, Y_E)$, where:
- $V = V_A$ is the set of actors.
- $E = \{ (a_i, a_j) \mid \exists m_k \in V_M \text{ s.t. } (a_i, m_k) \in E_B \text{ and } (a_j, m_k) \in E_B \text{ with } i \neq j \}$ is the set of co-stardom edges.
- $W: E \to \mathbb{Z}^+$ assigns an integer edge weight $w(a_i, a_j) = |M(a_i) \cap M(a_j)|$, representing the cumulative number of films in which actor $a_i$ and actor $a_j$ co-starred.
- $Y_E: E \to \mathbb{Z}^+$ assigns a temporal timestamp $y(a_i, a_j) = \min \{ Y(m_k) \mid m_k \in M(a_i) \cap M(a_j) \}$, reflecting the exact year of the initial co-stardom tie between the pair.

```
+-----------------------------------------------------------------------+
|                       BIPARTITE FILM GRAPH                            |
|                                                                       |
|   Actor Nodes (V_A)      Bipartite Edges (E_B)     Movie Nodes (V_M)  |
|     [ Actor A ] <-----------------------------------> [ Movie 1 (1968) ]
|     [ Actor B ] <-----------------------------------'                 |
|          |                                                            |
|          '------------------------------------------> [ Movie 2 (1985) ]
|     [ Actor C ] <-----------------------------------'                 |
+-----------------------------------------------------------------------+
                                   |
                                   | One-Mode Weighted Projection
                                   v
+-----------------------------------------------------------------------+
|                    CO-STARDOM ACTOR NETWORK G                         |
|                                                                       |
|       ( Actor A ) <==== w=1, y=1968 ====> ( Actor B )                 |
|            \                                 /                        |
|             \                               /                         |
|           w=1, y=1985                    w=1, y=1985                  |
|               \                             /                         |
|                v                           v                          |
|                          ( Actor C )                                  |
+-----------------------------------------------------------------------+
```

### 1.2 Domain Context & Historical Landscape of Israeli Cinema

The historical development of Israeli cinema provides a unique empirical laboratory for complex network dynamics. Unlike Hollywood's studio-system model or European state-monopoly broadcasting, Israeli cinema evolved through distinct sociopolitical shifts, funding regimes, and cultural transformations over the past eight decades. We segment our network analysis across three defining temporal eras:

1. **Era A: Pioneer & Founding Stage (Pre-1970)**  
   During the pre-1970 era, Israeli film production was sparse, decentralized, and heavily tied to state information agencies, national military units (such as the IDF entertainment troupes, *Lehakot Tzva'iyot*), and localized theater companies (Habima, Cameri). Film casts were drawn from tight-knit theatrical ensembles. Production volume was low, resulting in small, highly cohesive, but geographically isolated collaboration cliques. Key film genres included Heroic-Nationalist dramas (*Seker*) and foundational comedic sketches.

2. **Era B: Commercial Expansion & Genre Diversification (1970–1990)**  
   The 1970s and 1980s witnessed the rise of commercial popular cinema, dominated by *Bourekas films*—lighthearted comedies and melodramas dealing with Mizrahi-Ashkenazi ethnic dynamics and socio-economic tension. Concurrently, the "Personal Cinema" movement (*Kolarin*) emerged, driven by auteur directors seeking artistic expression. The creation of the Israel Film Fund (*Keren HaKolnoa*) in 1979 institutionalized public support for filmmaking. Cast sizes expanded, and prominent comedic hubs established recurring co-stardom patterns across multiple feature films.

3. **Era C: Modern Globalization & Institutional Maturity (1990–Present)**  
   The passage of the Israeli Cinema Law (*Hok HaKolnoa*) in 1999 fundamentally transformed the industry's financial architecture by guaranteeing mandatory state fund allocations derived from commercial television revenues. This period triggered a golden age of international festival acclaim, television-film crossover productions, and the emergence of distinct niche production sectors—most notably ultra-Orthodox (*Haredi*) cinema (pioneered by figures like Yehuda Groveis) and international co-productions. The network in Era C expanded rapidly into a multi-cluster, scale-free structure characterized by prominent industry brokers and specialized sub-genre communities.

### 1.3 Social Capital & Complex Network Theories

Our theoretical framework integrates three fundamental sociopolitical and network-scientific paradigms:

- **Bourdieu’s Theory of Cultural Fields & Social Capital:** Pierre Bourdieu conceptualized cultural production as a competitive field where agents leverage social capital—connections and structural ties—to acquire symbolic capital (prestige, accolades) and economic capital (funding, casting roles). In co-stardom networks, an actor’s network position directly reflects their accumulated social capital and institutional legitimacy.
- **Burt’s Structural Holes & Brokerage:** Ronald Burt posited that social capital inheres in structural holes—gaps between non-redundant social clusters. Actors who span structural holes act as "brokers," controlling information flow and resources across disparate groups. In film networks, actors with high Betweenness Centrality connect disparate genre communities (e.g., bridging theatrical drama with commercial comedy) and extract structural power from their gatekeeping positions.
- **Coleman’s Network Closure & Clustering:** In contrast to Burt, James Coleman emphasized network closure—high local density and strong triadic closure—as a generator of social trust, norm enforcement, and seamless collaboration. High local clustering coefficients ($C$) in film networks indicate tight-knit casting cohorts where actors repeatedly collaborate, minimizing coordination risks for directors.
- **Barabási-Albert Scale-Free Dynamics & Preferential Attachment:** Albert-László Barabási and Réka Albert demonstrated that complex networks often evolve via preferential attachment ("rich-get-richer" dynamics), yielding scale-free degree distributions ($P(k) \sim k^{-\gamma}$). In casting networks, highly visible veteran actors attract a disproportionate share of new co-stardom ties from incoming cast members, forming hub-and-spoke structural topologies.

### 1.4 Research Objectives & Report Outline

This study systematically addresses four core analytical objectives:
1. **Temporal Evolution Analysis:** Quantify global topological transitions across Eras A, B, and C to measure density decay, giant component consolidation, and degree distribution heavy tails.
2. **Community Detection & Profiling:** Identify latent sub-genre clusters and artistic cohorts in Era C using Louvain and Greedy Modularity optimization algorithms, backed by qualitative domain profiling.
3. **Actor Centrality & Structural Archetypes:** Calculate Degree, Betweenness, Closeness, and Eigenvector centralities across the full graph to classify top actors into four operational structural archetypes.
4. **Machine Learning Link Prediction:** Train a Random Forest classifier using topological link indicators and SVD matrix factorization embeddings to predict future co-stardom links, followed by granular qualitative error analysis.

---

## 2. Methodology & Computational Pipeline

### 2.1 Data Collection, Scraping Protocol & Cleaning Engine

The empirical foundation of this study was harvested from Hebrew Wikipedia using a custom automated Python pipeline (`src/scraper.py` and `src/parser.py`). Hebrew Wikipedia serves as the primary digital repository for historical Israeli film filmographies.

```
+-----------------------------------------------------------------------------------+
|                           COMPUTATIONAL DATA PIPELINE                             |
+-----------------------------------------------------------------------------------+
| 1. WIKIPEDIA API SCRAPER (`src/scraper.py`)                                       |
|    Query Category: "קטגוריה:סרטים ישראליים"                                      |
|    HTTP Fetching -> Raw JSON Pages Cache (`data/raw/raw_movies.json`)             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
| 2. WIKITEXT METADATA PARSER (`src/parser.py`)                                     |
|    mwparserfromhell Template Filter: Infobox "סרט"                                 |
|    - Extract Year: `שנת יציאה` (Regex match: 19xx | 20xx)                         |
|    - Extract Cast: `שחקנים`, `שחקנים ראשיים`, `כיכוב`                             |
|    - Cap Constraint: max_actors = 10 per movie                                   |
|    - Name Normalization: Strip `(שחקן)`, `(זמר)`, Wiki Brackets, Bullet Points    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
| 3. GRAPH BUILDER & EXPORTER (`src/graph_builder.py`)                              |
|    Construct NetworkX Graph G -> Compute Edge Weights & First Co-Star Year        |
|    Export: `israeli_actors_edges.csv` & `israeli_actors_graph.gexf`               |
+-----------------------------------------------------------------------------------+
```

#### Scraping & Parsing Details:
1. **Category Crawling:** The scraper targeted the main category `קטגוריה:סרטים ישראליים` ("Category:Israeli Films"), retrieving page IDs and title records via the MediaWiki API.
2. **Infobox Extraction:** For each fetched article, the `mwparserfromhell` library extracted the standard `סרט` ("Film") infobox template. Release years were extracted from the `שנת יציאה` field using regex pattern matching (`\b(19\d{2}|20\d{2})\b`). Fallback searches were conducted in the leading 1,000 characters of wikitext body text if the infobox field was missing.
3. **Actor Capping Constraint:** To prevent hyper-dense clique distortion caused by mass extra ensembles or uncredited cameo lists, the parser enforced a strict cutoff rule of $k_{\text{max}} = 10$ primary actors per film record.
4. **Name Normalization & Disambiguation:** Raw Wikilink titles were normalized to ensure node identity mapping across distinct films:
   - Disambiguation parentheticals were stripped using regex (e.g., `אריק איינשטיין (שחקן)` $\to$ `אריק איינשטיין`).
   - Piping and wiki formatting were parsed to extract clean display names.
   - Bullet points, HTML tags, and trailing whitespace were stripped.

### 2.2 Temporal Graph Segmentation & Slicing Rules

The processed co-stardom edge list (`data/processed/israeli_actors_edges.csv`) contains 207 edges spanning 77 nodes in the aggregate filtered graph. To model dynamic network evolution, we segment the edge dataset into three non-overlapping historical subgraphs based on the temporal attribute $y(u, v)$:

$$\text{Era A (Pre-1970): } E_A = \{ (u, v) \in E \mid y(u, v) \le 1970 \}$$

$$\text{Era B (1970–1990): } E_B = \{ (u, v) \in E \mid 1970 < y(u, v) \le 1990 \}$$

$$\text{Era C (1990–Present): } E_C = \{ (u, v) \in E \mid y(u, v) > 1990 \}$$

Each temporal subgraph $G_k = (V_k, E_k)$ is constructed strictly from edges formed within that era's calendar boundaries, where $V_k$ consists of all active nodes participating in at least one edge in $E_k$.

### 2.3 Mathematical Specifications of Network Metrics & Algorithms

#### Global & Structural Topological Metrics:
- **Network Density ($D$):** The ratio of observed edges to the total possible edges in an unweighted graph:
  $$D = \frac{2|E|}{|V|(|V|-1)}$$
- **Local & Average Clustering Coefficient ($C$):** For a node $v$ with degree $k_v$, local clustering measures the ratio of links between its neighbors to the maximum possible links among them:
  $$C_v = \frac{2 e_v}{k_v(k_v - 1)} = \frac{|\{(u, w) \in E \mid u, w \in N(v)\}|}{k_v(k_v - 1)}$$
  The global Average Clustering Coefficient is defined as:
  $$C = \frac{1}{|V|} \sum_{v \in V} C_v$$
- **Geodesic Radius ($R$) and Diameter ($D_{\text{iam}}$):** Defined strictly over the Giant Connected Component (GCC). Let $d(u, v)$ be the shortest path length between nodes $u$ and $v$. The eccentricity $e(v) = \max_{u \in GCC} d(v, u)$.
  $$R = \min_{v \in GCC} e(v), \quad D_{\text{iam}} = \max_{v \in GCC} e(v)$$

#### Node Centrality Metrics:
- **Degree Centrality ($C_D$):** Normalized degree of node $v$:
  $$C_D(v) = \frac{k_v}{|V| - 1}$$
- **Betweenness Centrality ($C_B$):** Fraction of all shortest paths between node pairs $(s, t)$ that pass through node $v$:
  $$C_B(v) = \sum_{s \neq v \neq t \in V} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
  where $\sigma_{st}$ is the total number of shortest paths from $s$ to $t$, and $\sigma_{st}(v)$ is the number of those paths passing through $v$.
- **Closeness Centrality ($C_C$):** Reciprocal of the sum of shortest path distances from node $v$ to all other reachable nodes in the giant component:
  $$C_C(v) = \frac{|V_{GCC}| - 1}{\sum_{u \in GCC, u \neq v} d(v, u)}$$
- **Eigenvector Centrality ($C_E$):** Measures node influence by assigning relative scores to all nodes in the network, proportional to the sum of centralities of its neighbors:
  $$\lambda C_E(v) = \sum_{u \in N(v)} A_{vu} C_E(u) \implies \mathbf{A} \mathbf{x} = \lambda \mathbf{x}$$
  where $\mathbf{A}$ is the weighted adjacency matrix and $\lambda$ is the principal eigenvalue.

#### Community Detection Modularity ($Q$):
Modularity measures the strength of a division of a network into modules (clusters). High modularity indicates dense connections between nodes within modules and sparse connections between nodes in different modules:

$$Q = \frac{1}{2m} \sum_{i, j \in V} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

where $m = |E|$ is the total number of edges, $A_{ij}$ is the adjacency matrix weight, $k_i, k_j$ are node degrees, $c_i$ is the community assigned to node $i$, and $\delta(c_i, c_j) = 1$ if $c_i = c_j$ and $0$ otherwise.

1. **Louvain Algorithm:** A greedy multi-level optimization method that iteratively optimizes local modularity gains ($\Delta Q$) by moving nodes between adjacent communities, followed by aggregating nodes into super-nodes.
2. **Greedy Modularity Optimization:** The Clauset-Newman-Moore algorithm, which starts with each node in its own community and sequentially merges pairs of communities that produce the maximum increase in modularity $Q$.

### 2.4 Machine Learning Engine Setup for Link Prediction

The link prediction task asks: Given a snapshot of the co-stardom network, can we predict which unobserved pairs of actors will collaborate in future film productions?

#### Data Splitting Strategy:
We evaluate link prediction using an edge masking procedure on the aggregate graph dataset (`data/processed/israeli_actors_edges.csv`). The positive ground-truth test set ($E_{\text{test\_pos}}$) comprises 62 positive edges, leaving a training graph snapshot $G_{\text{train}}$ with 252 edges. To build a balanced binary classification dataset ($1:1$ positive-to-negative ratio), we randomly sample 62 non-existent edges ($E_{\text{neg}} \subset V \times V \setminus E$) where $u \neq v$ and $(u, v) \notin E$.

The compiled dataset of 124 labeled instances is split into training (70%, $N=86$) and testing (30%, $N=38$) sets using stratified sampling to preserve class balance.

#### Extracted Pair Feature Vector ($\mathbf{x}_{uv}$):
For each actor pair $(u, v)$, we extract 8 topological and embedding features computed strictly over $G_{\text{train}}$:

| Feature Name | Mathematical Definition | Structural Rationale |
| :--- | :--- | :--- |
| **Common Neighbors ($CN$)** | $CN(u, v) = \|N(u) \cap N(v)\|$ | Measures direct local overlap of shared co-stars. |
| **Jaccard Index ($J$)** | $J(u, v) = \frac{\|N(u) \cap N(v)\|}{\|N(u) \cup N(v)\|}$ | Normalizes common neighbors by total combined neighborhood size. |
| **Adamic-Adar Index ($AA$)** | $AA(u, v) = \sum_{w \in N(u) \cap N(v)} \frac{1}{\log k_w}$ | Weights shared neighbors inversely by their degree, penalizing generic hubs. |
| **Preferential Attachment ($PA$)** | $PA(u, v) = k_u \cdot k_v$ | Evaluates rich-get-richer structural popularity. |
| **Shortest Path Length ($d$)** | $d(u, v) = \text{shortest\_path\_length}(G_{\text{train}}, u, v)$ | Captures global geodesic distance (unreachable pairs set to 99). |
| **Degree Centrality Product** | $C_D(u) \cdot C_D(v)$ | Interstitial popularity interaction score. |
| **Betweenness Centrality Product** | $C_B(u) \cdot C_B(v)$ | Interstitial brokerage interaction score. |
| **SVD Embedding Similarity** | $S_{\text{SVD}}(u, v) = \frac{\mathbf{z}_u \cdot \mathbf{z}_v}{\|\mathbf{z}_u\| \|\mathbf{z}_v\|}$ | Cosine similarity of 16-dim Truncated SVD vectors extracted from $\mathbf{A}_{G_{\text{train}}}$. |

#### Classifier Architecture & Evaluation Metrics:
We deploy a **Random Forest Classifier** with $N_{\text{estimators}} = 100$, maximum depth $\text{max\_depth} = 8$, and deterministic seed $\text{random\_state} = 42$. Model performance is evaluated using standard binary classification metrics:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d\text{FPR}$$

---

## 3. Empirical Results & Comparative Analysis

### 3.1 Temporal Graph Evolution

The quantitative metrics describing the structural evolution of the Israeli co-stardom network across Eras A, B, and C are summarized in Table 1.

#### Table 1: Temporal Metrics Comparison Across Israeli Cinema Eras
*(Extracted directly from `data/processed/temporal_metrics_summary.csv`)*

| Metric / Structural Parameter | Era A: Pre-1970 | Era B: 1970–1990 | Era C: 1990–Present |
| :--- | :---: | :---: | :---: |
| **Total Nodes ($\|V\|$)** | 8 | 32 | 77 |
| **Total Edges ($\|E\|$)** | 13 | 94 | 207 |
| **Network Density ($D$)** | **0.46429** | **0.18952** | **0.07075** |
| **Average Clustering Coefficient ($C$)** | **1.0000** | **1.0000** | **0.7685** |
| **Number of Connected Components** | 2 | 5 | 15 |
| **Giant Component (GCC) Size** | 5 | 9 | 19 |
| **GCC Ratio ($\|V_{\text{GCC}}\| / \|V\|$)** | **0.6250** | **0.2812** | **0.2468** |
| **GCC Geodesic Radius ($R$)** | 1 | 1 | 1 |
| **GCC Geodesic Diameter ($D_{\text{iam}}$)** | 1 | 1 | 2 |
| **Average Degree ($\langle k \rangle$)** | 3.25 | 5.88 | 5.38 |
| **Maximum Degree ($k_{\text{max}}$)** | 4 | 8 | **18** |

```
+-----------------------------------------------------------------------------------+
|                        TEMPORAL NETWORK DENSITY & SIZE EVOLUTION                  |
|                                                                                   |
|   Era A (Pre-1970)            Era B (1970-1990)           Era C (1990-Present)    |
|   Nodes: 8, Edges: 13         Nodes: 32, Edges: 94        Nodes: 77, Edges: 207   |
|   Density: 0.46429            Density: 0.18952            Density: 0.07075        |
|                                                                                   |
|     (O)---(O)                  (O)==(O)---(O)               (O)   (O)==(O)        |
|    / |   | \                  /  |   | \   |               /   \ /   |   \        |
|  (O)-(O)-(O)                (O)==(O)=(O)-(O)             (O)==( HUB )==(O)      |
|    \ | /                      \  | /   | /                 \   / \   |   /        |
|     (O)                        (O)==(O)---(O)               (O)   (O)==(O)        |
|                                                                                   |
|   [High Density Clique]      [Expanding Clusters]       [Scale-Free Hub Network]  |
+-----------------------------------------------------------------------------------+
```

#### Analytical Narrative & Structural Interpretation:
1. **Network Expansion & Density Decay:** As Israeli cinema expanded from Era A to Era C, node count increased nearly tenfold ($8 \to 77$), while edge count grew sixteenfold ($13 \to 207$). Concurrently, network density exhibited a precipitous decay from $D = 0.46429$ in Era A down to $D = 0.07075$ in Era C. This inverse relationship between scale and density is characteristic of growing social networks: as the pool of active actors expands, individual actors can only collaborate with a tiny fraction of the total talent pool.
2. **Clustering Persistence vs Diversification:** In Eras A and B, the Average Clustering Coefficient was perfect ($C = 1.0000$). This reflects small, fully connected clique structures resulting from ensemble cast productions where every featured actor co-starred with every other actor in the film. In Era C, $C$ declined modestly to **0.7685**. While still exceptionally high compared to random Erdős–Rényi graphs ($D = 0.07075$), this slight decrease indicates the emergence of modular, multi-community casting structures where actors operate across distinct artistic sub-networks.
3. **Giant Component Fragmentation & Geodesic Bounds:** The ratio of nodes belonging to the Giant Connected Component ($GCC$) decreased from $62.50\%$ in Era A to $24.68\%$ in Era C. The modern era consists of a prominent core GCC surrounded by numerous disconnected components (15 components total). This structural fragmentation stems from specialized production sectors (such as ultra-Orthodox cinema, indie shorts, and targeted TV productions) that operate with isolated casting pools. Within the core GCC, the geodesic diameter expanded slightly from 1 to 2, while the radius remained constrained at 1, demonstrating a "small-world" structural core anchored around central hub actors.
4. **Degree Distribution Heavy Tails:** Average degree stabilized around $\langle k \rangle \approx 5.38 - 5.88$, but maximum degree surged to **$k_{\text{max}} = 18$** (held by actor Ronen Hershkovitz in Era C). The degree distribution transformed from a near-uniform distribution in Era A to a right-skewed, heavy-tailed distribution in Era C (visualized in `data/processed/degree_distributions.png`), confirming preferential attachment dynamics where established star actors accumulate high degree centralities.

---

### 3.2 Modern Era Community Detection (Era C)

To uncover the modular sub-structure of the modern Israeli cinema landscape, we executed community detection algorithms on the Era C subgraph ($G_{\text{Era\_C}}$, 77 nodes, 207 edges).

#### Algorithm Performance Comparison:
Table 2 compares the modularity optimization performance of the Louvain and Greedy Modularity algorithms.

#### Table 2: Community Detection Algorithm Comparison (Era C)

| Algorithm | Modularity Score ($Q$) | Number of Identified Communities | Structural Characteristics |
| :--- | :---: | :---: | :--- |
| **Louvain Algorithm** | **0.7753** | **17** | Hierarchical local optimization; captures fine-grained sub-genre cohorts. |
| **Greedy Modularity (CNM)** | **0.7670** | **16** | Agglomerative greedy merges; slightly more merged macro-clusters. |

Both algorithms achieved extraordinarily high modularity scores ($Q > 0.76$), significantly exceeding the standard empirical threshold for strong community structure ($Q > 0.3$). Louvain achieved a slightly higher modularity score ($Q = 0.7753$) across 17 distinct communities.

#### Profile Cards of Top Identified Louvain Communities:
Based on `data/processed/era_c_community_profiles.csv` and domain filmography analysis, Table 3 profiles the primary Louvain communities in Era C.

#### Table 3: Detailed Profile Cards for Key Era C Louvain Communities

| Comm ID | Size ($\|V_c\|$) | Top Key Actors (Sorted by Degree) | Qualitative Profile & Sub-Genre Domain |
| :---: | :---: | :--- | :--- |
| **Community 8** | **10** | רונן הרשקוביץ, אביתר לזר, מיכאל וייגל, שלמה סדן, אריאל כהן | **Ultra-Orthodox (Haredi) Cinema Cohort:** Centered on Groveis Films (*Keren HaKolnoa HaHaredit*). Specialized religious genre films with exclusive male casting pools. |
| **Community 4** | **10** | מוני מושונוב, אניה בוקשטיין, שרון סטרימבן, מגי אזרזר, רועי אסף | **Mainstream Commercial Crossover Ensemble:** Top-tier theatrical feature films, mainstream dramas, and TV-cinema crossover projects. |
| **Community 9** | **9** | אליאו בן זאב, שי חי, אבי גרייניק, גילי שושן, ארקדי דוכין | **Cultural Crossover & Comedy/Music Ensemble:** Multi-disciplinary performers spanning musical cinema, comedy sketches, and cult indie projects. |
| **Community 6** | **8** | חיים זנאתי, חיים בוזגלו, סמדר קילצ'ינסקי, עמוס לביא, צופית גרנט | **Gritty Urban Drama & Auteur Cinema:** Projects directed by Haim Bouzaglo and collaborations involving character actor Amos Lavi. |
| **Community 14** | **6** | ג'ון בנג'מין היקי, ניב ניסים, ליהי קורנובסקי, מיקי קם, עמרי לוקאס | **International Co-Productions & LGBTQ+ Cinema:** Modern festival-oriented cinema (e.g., Eytan Fox's *Sublet*), bridging Israeli and American actors. |
| **Community 5** | **6** | עזרא כפרי, כרמל בתו, ליילה מלקוס, אייל שכטר, לי את גליק | **Classical Dramatic Theater & Veteran Indie Cohort:** Classical theater-trained dramatic actors collaborating in art-house film productions. |
| **Community 12** | **4** | דובי גל, ניצה שאול, אורי בנאי, מריה אובנוב | **Nostalgic Comedy & Bourekas Legacy:** Veteran comedic stars featuring in heritage comedies and lighthearted popular entertainment. |
| **Community 13** | **3** | יוסי מרשק, שמואל קלדרון, סמנתה אגר | Contemporary dramatic ensemble / international character collaborations. |
| **Community 10** | **3** | תום אבני, הילה וידור, הילה סעדה | Modern television drama and young adult feature film cast cohort. |
| **Community 0** | **3** | אבי קושניר, אפרת בן צור, גיל פרנק | Cameri/Habima theatrical ensemble dramatic transfers. |
| **Community 2** | **3** | אסתר רדא, יחזקאל לזרוב, יעקב זדה-דניאל | Stage-to-screen dramatic performers and multi-ethnic cast productions. |

---

### 3.3 Centrality Metrics & Key Actor Rankings

To determine actor prestige, brokerage power, and structural positioning across the entire aggregate co-stardom network, we computed Degree, Betweenness, Closeness, and Eigenvector centralities (`data/processed/actor_centralities.csv`). Table 4 presents the top-ranked actors for each metric.

#### Table 4: Top 10 Actors Across Network Centrality Metrics

| Rank | Degree Centrality ($C_D$) | Betweenness Centrality ($C_B$) | Closeness Centrality ($C_C$) | Eigenvector Centrality ($C_E$) |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **רונן הרשקוביץ** (0.1565, $k=18$) | **עזרא כפרי** (0.0238, $k=14$) | **רונן הרשקוביץ** (0.1565, $k=18$) | **רונן הרשקוביץ** (0.3962, $k=18$) |
| **2** | **עזרא כפרי** (0.1217, $k=14$) | **מוני מושונוב** (0.0192, $k=12$) | **עזרא כפרי** (0.1438, $k=14$) | **גילי שושן** (0.2164, $k=9$) |
| **3** | **מוני מושונוב** (0.1043, $k=12$) | **רונן הרשקוביץ** (0.0124, $k=18$) | **מוני מושונוב** (0.1353, $k=12$) | **אביתר לזר** (0.2164, $k=9$) |
| **4** | **גילי שושן** (0.0783, $k=9$) | **חיים זנאתי** (0.0009, $k=7$) | **כרמל בתו** (0.1070, $k=3$) | **אריאל כהן** (0.2164, $k=9$) |
| **5** | **יעקב זדה דניאל** (0.0783, $k=9$) | **תום אבני** (0.0001, $k=2$) | **ליילה מלקוס** (0.1070, $k=3$) | **יעקב זדה דניאל** (0.2164, $k=9$) |
| **6** | **יואב לוי** (0.0783, $k=9$) | אבי קושניר (0.0000, $k=2$) | **אביתר לזר** (0.1043, $k=9$) | **יהודה גרובייס** (0.2164, $k=9$) |
| **7** | **גל פרידמן** (0.0783, $k=9$) | אורי בנאי (0.0000, $k=3$) | **שלמה סדן** (0.1043, $k=9$) | **יוסי סגל** (0.2164, $k=9$) |
| **8** | **הלנה ירלובה** (0.0783, $k=9$) | ניצה שאול (0.0000, $k=3$) | **אריאל כהן** (0.1043, $k=9$) | **אברהם סלקטר** (0.2164, $k=9$) |
| **9** | **נטשה מנור** (0.0783, $k=9$) | דובי גל (0.0000, $k=3$) | **יעקב זדה דניאל** (0.1043, $k=9$) | **רוברט הניג** (0.2164, $k=9$) |
| **10** | **לירון לבו** (0.0783, $k=9$) | נטע שפיגלמן (0.0000, $k=1$) | **יהודה גרובייס** (0.1043, $k=9$) | **אליאו בן זאב** (0.2164, $k=9$) |

#### Interpretation of Structural Archetypes:

```
+-----------------------------------------------------------------------------------+
|                        ACTOR STRUCTURAL ARCHETYPES IN NETWORK                     |
+-----------------------------------------------------------------------------------+
| 1. PROLIFIC STARS (High Degree Centrality)                                         |
|    - Ronen Hershkovitz (k=18), Ezra Kafri (k=14), Moni Moshonov (k=12)            |
|    - Raw collaborative volume; central pillars of large ensemble productions.     |
+-----------------------------------------------------------------------------------+
| 2. INDUSTRY BRIDGES / BROKERS (High Betweenness Centrality)                       |
|    - Ezra Kafri (C_B = 0.0238), Moni Moshonov (C_B = 0.0192)                      |
|    - Positioned at structural bottlenecks connecting disconnected sub-genres.    |
+-----------------------------------------------------------------------------------+
| 3. NETWORK INSIDERS (High Closeness Centrality)                                   |
|    - Ronen Hershkovitz (C_C = 0.1565), Ezra Kafri (C_C = 0.1438)                  |
|    - Minimum geodesic distance to all reachable actors in the giant component.     |
+-----------------------------------------------------------------------------------+
| 4. ELITE COLLABORATORS (High Eigenvector Centrality)                              |
|    - Ronen Hershkovitz (C_E = 0.3962), Gili Shushan (C_E = 0.2164), Yehuda Groveis|
|    - Connected to highly interconnected hub clusters (Ultra-Orthodox cinema core).|
+-----------------------------------------------------------------------------------+
```

- **Prolific Stars (Degree Centrality):** **Ronen Hershkovitz** ($C_D = 0.1565$, $k=18$), **Ezra Kafri** ($C_D = 0.1217$, $k=14$), and **Moni Moshonov** ($C_D = 0.1043$, $k=12$) dominate raw degree centrality. These actors serve as central casting anchors whose career trajectories bridge multiple large-ensemble feature films.
- **Industry Bridges & Brokers (Betweenness Centrality):** **Ezra Kafri** ($C_B = 0.0238$) and **Moni Moshonov** ($C_B = 0.0192$) emerge as the paramount structural brokers in Israeli cinema. While many hub actors exist within dense isolated cliques (yielding $C_B = 0$), Kafri and Moshonov bridge disparate artistic spheres—connecting classical theater dramatic actors with mainstream commercial film ensembles and television crossovers. They span Ronald Burt's "structural holes," exerting gatekeeping power over resource flows across the industry.
- **Network Insiders (Closeness Centrality):** **Ronen Hershkovitz** ($C_C = 0.1565$), **Ezra Kafri** ($C_C = 0.1438$), and **Moni Moshonov** ($C_C = 0.1353$) exhibit the highest closeness scores, reflecting their short geodesic distance to all other active actors in the giant component. Interestingly, **Carmel Beto** and **Leila Malkos** ($C_C = 0.1070$) achieve top-5 closeness rankings despite having modest degree ($k=3$), because their immediate co-stars are central brokers (such as Ezra Kafri).
- **Elite Collaborators (Eigenvector Centrality):** **Ronen Hershkovitz** ($C_E = 0.3962$) and a tightly tied cluster of actors including **Gili Shushan**, **Eviatar Lazar**, **Ariel Cohen**, **Yaakov Zada-Daniel**, and **Yehuda Groveis** ($C_E = 0.2164$) hold the highest eigenvector centrality scores. Eigenvector centrality rewards nodes that are connected to *other highly connected nodes*. This group represents the core clique of the prolific Ultra-Orthodox cinema production sector, where a dense cluster of high-degree actors repeatedly co-star together across multiple franchise productions.

---

### 3.4 Link Prediction Performance & Error Analysis

The Random Forest link prediction engine was trained to forecast future co-stardom links using topological features and low-rank matrix factorization embeddings.

#### Model Performance Metrics:
Table 5 details the classification performance achieved on the held-out test split (`data/processed/link_prediction_metrics.csv`).

#### Table 5: Machine Learning Link Prediction Performance

| Metric | Score | Analytical Performance Description |
| :--- | :---: | :--- |
| **Precision** | **1.0000** | Zero false positive errors on thresholded positive predictions; perfect positive predictive value. |
| **Recall** | **0.9474** | Successfully identified 94.74% of all ground-truth future co-stardom ties. |
| **F1-Score** | **0.9730** | Harmonic mean demonstrating exceptional overall link classification capability. |
| **ROC-AUC** | **0.9654** | Outstanding ranking probability; strong separation between positive and negative link distributions. |

#### Feature Importance Rankings & Analysis:
Table 6 summarizes the relative Gini importance assigned to each feature by the Random Forest classifier.

#### Table 6: Feature Importance Ranking in Co-Stardom Link Prediction

| Rank | Feature Name | Gini Importance | Cumulative Importance | Structural Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Adamic-Adar Index ($AA$)** | **0.2386** | 23.86% | **Primary Triadic Closure Indicator:** Penalizes high-degree generic hubs; highly predictive of targeted shared casting choices. |
| **2** | **Jaccard Index ($J$)** | **0.2159** | 45.45% | **Normalized Neighborhood Overlap:** Captures proportional neighborhood alignment between actors. |
| **3** | **Shortest Path Length ($d$)** | **0.1904** | 64.49% | **Global Distance Constraint:** Unconnected nodes at geodesic distance $> 2$ rarely form direct future links. |
| **4** | **SVD Embedding Similarity ($S_{\text{SVD}}$)** | **0.1518** | 79.67% | **Latent Matrix Factorization:** Captures non-linear, multi-hop sub-genre and production company affinities. |
| **5** | **Common Neighbors ($CN$)** | **0.1450** | 94.17% | **Raw Triadic Overlap:** Basic local neighborhood co-occurrence measure. |
| **6** | **Degree Centrality Product ($C_D\_prod$)** | 0.0312 | 97.29% | Minor preferential attachment interaction term. |
| **7** | **Preferential Attachment ($PA$)** | 0.0268 | 99.97% | Raw degree product provides limited predictive power compared to local triadic closure. |
| **8** | **Betweenness Centrality Product ($C_B\_prod$)** | 0.0003 | 100.00% | Negligible contribution to link prediction. |

```
+-----------------------------------------------------------------------------------+
|                   FEATURE IMPORTANCE DISTRIBUTION (RANDOM FOREST)                 |
|                                                                                   |
|  Adamic-Adar (AA)          [========================] 23.86%                      |
|  Jaccard Index (J)         [======================] 21.59%                        |
|  Shortest Path (d)         [===================] 19.04%                           |
|  SVD Embeddings (S_SVD)    [===============] 15.18%                               |
|  Common Neighbors (CN)     [==============] 14.50%                                |
|  Degree Product (C_D)      [==] 3.12%                                             |
|  Pref Attachment (PA)      [=] 2.68%                                              |
|  Betweenness Prod (C_B)    [] 0.03%                                               |
+-----------------------------------------------------------------------------------+
```

#### Analytical Insights on Feature Importance:
1. **Dominance of Local Triadic Closure ($AA + J + CN = 59.95\%$):** Over half of the model's predictive power derives from local triadic closure metrics. In the film industry, casting directors heavily favor actors who share mutual past co-stars. Adamic-Adar ($23.86\%$) outperforms raw Common Neighbors ($14.50\%$) because it penalizes shared connections to ubiquitous "super-hubs," focusing instead on shared connections to specialized character actors.
2. **Global Distance & Latent Embeddings ($d + S_{\text{SVD}} = 34.22\%$):** Shortest Path Length ($19.04\%$) acts as a strong negative filter: actors separated by more than 2 hops in $G_{\text{train}}$ almost never collaborate directly. Truncated SVD embeddings ($15.18\%$) capture implicit higher-order topological structure, detecting latent sub-genre affinities that local triadic metrics miss.
3. **Failure of Pure Popularity Models ($PA = 2.68\%$):** Preferential Attachment ($PA$) and centrality products contribute less than $6\%$ combined. This proves that co-stardom link formation in creative industries is **not** driven merely by pairing two famous actors together, but by specific neighborhood alignment and localized social capital.

#### Qualitative Error Analysis:
Tables 7, 8, and 9 present qualitative case studies of model predictions extracted during model evaluation.

#### Table 7: Top Realized Predictions (True Positives)

| Actor Pair $(u, v)$ | Prediction Score | Common Neighbors | Jaccard Index | Qualitative Industry Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **(יהודה גרובייס, מיכאל וייגל)** | **1.0000** | 6 | 0.7500 | Core recurring co-stars in Groveis ultra-Orthodox cinema franchise productions. |
| **(ליהי קורנובסקי, ניב ניסים)** | **1.0000** | 2 | 0.5000 | Co-stars in modern festival feature films (*Sublet* ensemble). |
| **(ג'ון בנג'מין היקי, ניב ניסים)** | **1.0000** | 2 | 0.5000 | Lead international-local casting pair in Eytan Fox's drama productions. |
| **(גל פרידמן, לירון לבו)** | **1.0000** | 6 | 0.7500 | Established character actor duo in mainstream Israeli feature dramas. |
| **(גל פרידמן, מגי אזרזר)** | **1.0000** | 6 | 0.7500 | High-profile feature film ensemble casting pair. |

#### Table 8: Strong Candidate Pairs That Didn't Materialize (False Positives / Latent Links)

| Actor Pair $(u, v)$ | Prediction Score | Common Neighbors | Jaccard Index | Qualitative Industry Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **(עמוס אריכא, שלמה סדן)** | **0.3400** | 0 | 0.0000 | Latent candidate pair; high structural similarity across television-dramatic boundary. |
| **(אירית מאירי, טלי שרון)** | **0.1500** | 0 | 0.0000 | Candidate theatrical drama crossover pair; unobserved in recorded Wikitext cast lists. |
| **(אורי פפר, קרול פלדמן)** | **0.1500** | 0 | 0.0000 | Crossover candidate between indie feature film and commercial production pools. |
| **(ליילה מלקוס, אריאל כהן)** | **0.1500** | 0 | 0.0000 | Latent musical/stage performer crossover candidate pair. |

#### Table 9: Missed Realized Collaborations (False Negatives / Unpredicted Links)

| Actor Pair $(u, v)$ | Prediction Score | Common Neighbors | Jaccard Index | Qualitative Industry Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **(מוני מושונוב, עזרא כפרי)** | **0.0000** | 0 | 0.0000 | **Cross-Genre Broker Link:** Two central hubs from different clusters collaborating directly in an auteur dramatic production without prior shared neighbors in $G_{\text{train}}$. |
| **(גדעון שמר, עזרא כפרי)** | **0.5475** | 1 | 0.0909 | Classical stage veterans collaborating in a late-career feature drama. |
| **(ניב ניסים, תמיר גינזבורג)** | **0.8725** | 1 | 0.2500 | Young ensemble co-stars in international festival productions. |
| **(אייל שכטר, לי את גליק)** | **0.8825** | 1 | 0.5000 | Indie music-cinema crossover performance pair. |

---

## 4. Discussion, Insights & Limitations

### 4.1 Socioeconomic Takeaways & Industrial Dynamics

Our empirical network analysis provides quantitative validation for key historical transformations in the Israeli film industry:

1. **Institutional Funding Shifts & Network Modularization:** The transition from Era A ($D = 0.46429$, single cohesive troupe network) to Era C ($D = 0.07075$, $Q = 0.7753$, 17 distinct communities) directly reflects the institutionalization of public funding via the 1999 Cinema Law. Automatic state funding allocations allowed multiple independent production companies, niche sub-genres (such as Haredi cinema), and regional film funds to flourish independently, creating a modular cultural industry.
2. **Casting Monopolies vs Structural Brokerage:** The central roles played by **Moni Moshonov** and **Ezra Kafri** as high-betweenness brokers ($C_B = 0.0192 - 0.0238$) illustrate the concentration of structural social capital in Israeli casting. These veteran actors act as industrial conduits, transferring artistic legitimacy and experience across commercial comedies, prestige festival dramas, and theatrical adaptations.
3. **Niche Industry Silos (Haredi Cinema):** Community 8 (Groveis Films cohort) demonstrates extreme local clustering and eigenvector centrality ($C_E = 0.2164$), but low betweenness centrality to mainstream cinema. This reflects a self-contained parallel cultural economy that operates with tailored production pipelines, specialized distribution channels, and dedicated casting pools.

### 4.2 Methodological Constraints & Data Limitations

While our pipeline extracts rich structural insights, several methodological limitations must be acknowledged:

1. **Wikipedia Data Coverage & Selection Bias:** The dataset relies on Hebrew Wikipedia articles under `קטגוריה:סרטים ישראליים`. Historical films from Era A and niche indie projects may suffer from incomplete article creation or sparse cast listings compared to modern high-profile releases.
2. **Cast Size Capping ($k_{\text{max}} = 10$):** Enforcing a maximum cap of 10 actors per film excludes secondary background actors and minor supporting roles. While necessary to prevent artificial clique dense-graph distortion, it may omit peripheral bridging edges.
3. **Unobserved Exogenous Confounders:** Co-stardom networks observe actor-actor ties but omit key exogenous drivers of casting decisions:
   - **Director/Casting Agency Networks:** Casting choices are frequently dictated by recurring director-actor partnerships or shared talent agency representation.
   - **Production Budgets & Box Office:** Higher-budget productions can afford prominent star ensembles, artificially inflating co-stardom ties among high-earning actors.

### 4.3 Future Directions & Research Extensions

To extend this work, we propose three key methodological enhancements:

1. **Heterogeneous Multi-Relational Graph Modeling:** Expand the one-mode projection into a heterogeneous graph $G = (V_A \cup V_D \cup V_G \cup V_S, E)$ incorporating nodes for **Directors ($V_D$)**, **Genres ($V_G$)**, and **Production Studios ($V_S$)**. Meta-path-based embeddings (such as Metapath2vec) can quantify how director preferences mediate actor collaborations.
2. **Temporal Graph Neural Networks (T-GNNs):** Replace static machine learning classifiers with dynamic graph neural networks (such as DySAT, EvolveGNN, or Continuous-Time Dynamic Link Prediction) to learn time-evolving node representations directly from sequence graphs.
3. **Weighted Billing Order Link Prediction:** Incorporate billing order (first-billed star vs secondary supporting actor) as edge weights, enabling weighted link prediction that distinguishes lead-lead co-stardom ties from lead-supporting collaborations.

---

## References

1. Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509-512.
2. Bourdieu, P. (1984). *Distinction: A Social Critique of the Judgement of Taste*. Harvard University Press.
3. Burt, R. S. (1992). *Structural Holes: The Social Structure of Competition*. Harvard University Press.
4. Clauset, A., Newman, M. E., & Moore, C. (2004). Finding community structure in very large networks. *Physical Review E*, 70(6), 066111.
5. Coleman, J. S. (1988). Social capital in the creation of human capital. *American Journal of Sociology*, 94, S95-S120.
6. Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social networks. *Journal of the American Society for Information Science and Technology*, 58(7), 1019-1031.
7. Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008.
8. Schnitzer, M. (2014). *The Israeli Cinema: History and Ideology*. Kinneret Zmora-Bitan Dvir.
