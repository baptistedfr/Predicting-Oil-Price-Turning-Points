# Predicting-Oil-Turning-Points
Replication of the research paper : "**The prediction of oil price turning points with log-periodic power law and multi-population genetic algorithm**" _Fangzheng Chenga, Tijun Fana, Dandan Fanb, Shanling Li, 2018_

```Mermaid
flowchart TD
    %% Data & Configuration Layer
    subgraph "Data & Config"
        DataCSV["CSV Data (data)"]:::data
        Params["Parameter Config (params)"]:::config
    end

    %% Core Processing & Modeling Layer
    subgraph "Core Processing"
        AssetProcessor["AssetProcessor (GQLib/AssetProcessor.py)"]:::core
        Framework["Framework (GQLib/Framework.py)"]:::core
        LombAnalysis["LombAnalysis (GQLib/LombAnalysis.py)"]:::core
        subgraph "Models"
            LPPL["LPPL Model (GQLib/Models/LPPL.py)"]:::core
            LPPLS["LPPLS Model (GQLib/Models/LPPLS.py)"]:::core
        end
        Optimizers["Optimizers (GQLib/Optimizers)"]:::core
        JIT["JIT Functions (GQLib/njitFunc.py)"]:::core
        Enums["Enumerations (GQLib/enums.py)"]:::core
    end

    %% Execution Layer
    subgraph "Execution"
        Strat["strat.py"]:::exec
        Test["test.py"]:::exec
        Notebook["Etude_USO_WTI.ipynb"]:::exec
    end

    %% Output Layer
    subgraph "Output"
        Results["Results Output (Results)"]:::output
        ResultsStrategy["Strategy Results (Results_strategy)"]:::output
    end

    %% Documentation Layer
    subgraph "Documentation"
        Docs["Documentation (Docs)"]:::docs
    end

    %% Data Flow Connections
    DataCSV -->|"reads"| AssetProcessor
    Params -->|"configures"| AssetProcessor
    AssetProcessor -->|"orchestrates"| Framework
    Framework -->|"calls"| LPPL
    Framework -->|"calls"| LPPLS
    Framework -->|"selects"| Optimizers
    Framework -->|"analyzes"| LombAnalysis
    Optimizers -->|"accelerates via"| JIT
    Framework -->|"outputs"| Results
    Framework -->|"outputs"| ResultsStrategy
    Strat -->|"triggers"| Framework
    Test -->|"triggers"| Framework
    Notebook -->|"triggers"| Framework

    %% Click Events
    click DataCSV "https://github.com/baptistedfr/predicting-oil-price-turning-points/tree/main/data"
    click Params "https://github.com/baptistedfr/predicting-oil-price-turning-points/tree/main/params"
    click AssetProcessor "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/GQLib/AssetProcessor.py"
    click Framework "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/GQLib/Framework.py"
    click LombAnalysis "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/GQLib/LombAnalysis.py"
    click LPPL "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/GQLib/Models/LPPL.py"
    click LPPLS "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/GQLib/Models/LPPLS.py"
    click Optimizers "https://github.com/baptistedfr/predicting-oil-price-turning-points/tree/main/GQLib/Optimizers"
    click JIT "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/GQLib/njitFunc.py"
    click Enums "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/GQLib/enums.py"
    click Strat "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/strat.py"
    click Test "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/test.py"
    click Notebook "https://github.com/baptistedfr/predicting-oil-price-turning-points/blob/main/Etude_USO_WTI.ipynb"
    click Results "https://github.com/baptistedfr/predicting-oil-price-turning-points/tree/main/Results"
    click ResultsStrategy "https://github.com/baptistedfr/predicting-oil-price-turning-points/tree/main/Results_strategy"
    click Docs "https://github.com/baptistedfr/predicting-oil-price-turning-points/tree/main/Docs"

    %% Styles
    classDef data fill:#f9e79f,stroke:#e67e22,stroke-width:2px;
    classDef config fill:#f5cba7,stroke:#d35400,stroke-width:2px;
    classDef core fill:#aed6f1,stroke:#2e86c1,stroke-width:2px;
    classDef exec fill:#abebc6,stroke:#27ae60,stroke-width:2px;
    classDef output fill:#d5f5e3,stroke:#229954,stroke-width:2px;
    classDef docs fill:#fadbd8,stroke:#c0392b,stroke-width:2px;
```
