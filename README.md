# Taxo-Net

Este repositorio contiene el código y los modelos relacionados con Taxo-Net.

## Modos de Operación

*   **Entrenamiento (Train):** Este proceso toma el conjunto de datos de imágenes y ajusta los pesos de la red neuronal mediante iteraciones. Está diseñado para que la arquitectura aprenda a identificar características clave de las distintas categorías jerárquicas taxonómicas (familia, género, especie).
*   **Inferencia (Inference):** Una vez que el modelo ha sido entrenado, la inferencia se encarga de cargar los pesos resultantes para predecir sobre nuevas imágenes que el modelo nunca ha visto. Este proceso recibe una imagen y produce su correspondiente predicción taxonómica.

## Notas Importantes

1. **Rutas (Paths) en el Notebook de Entrenamiento:** El notebook para el entrenamiento de los modelos fue entrenado y ejecutado originalmente en una máquina Linux. Por lo tanto, si deseas ejecutarlo, asegúrate de que las **rutas de los archivos y directorios** sean actualizadas para que funcionen correctamente según tu sistema operativo local (Windows, macOS o Linux).
2. **Rutas en el Modelo:** De manera similar, si ejecutas el modelo o la inferencia, asegúrate de modificar todas las rutas en el código para que sean compatibles con el sistema operativo (SO) que estés usando, ya que el modelo también fue desarrollado en Linux.
3. **Descarga del Modelo Entrenado:** El modelo ya entrenado en estas condiciones se encuentra disponible para su descarga en el siguiente enlace de Google Drive:
   - 🔗 [Descargar Modelo Taxo-Net (Google Drive)](https://drive.google.com/file/d/1rBmSa5VPeBnj02JPi1QTVeouMM7Df4Mm/view?usp=drive_link)

## Arquitectura del Modelo (TaxoNet_ResNet50)

La arquitectura base del modelo consta de la extracción de características mediante ResNet-50 y tres derivaciones (cabezas de clasificación) independientes encargadas de realizar predicciones estructuradas:

```mermaid
graph TD
    A(["🖼️ Imagen de Entrada"]) --> B["Extracción de Características<br>(ResNet-50 Base sin capa final)"]
    B --> C["Capa Flatten<br>(Aplanar mapas de características)"]
    C --> D["Capa Dropout<br>(Regularización, p=0.4)"]
    
    D --> E["Cabeza de Familia<br>(Capa Lineal)"]
    D --> F["Cabeza de Género<br>(Capa Lineal)"]
    D --> G["Cabeza de Especie<br>(Capa Lineal)"]
    
    E --> H(["📊 Salida: Predicción de Familia"])
    F --> I(["📊 Salida: Predicción de Género"])
    G --> J(["📊 Salida: Predicción de Especie"])

    classDef input fill:#f4f4f4,stroke:#666,stroke-width:2px;
    classDef resnet fill:#ffe0b2,stroke:#f57c00,stroke-width:2px,color:#333;
    classDef linear fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#333;
    classDef output fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#333;
    
    class A input;
    class B,C,D resnet;
    class E,F,G linear;
    class H,I,J output;
```

## Función de Pérdida (TaxonomicPenaltyLoss)

Esta red es guiada por una métrica de pérdida unificada, donde los errores de clasificación en Familia, Género y Especie son castigados individualmente según ciertos pesos multiplicadores:

```mermaid
graph TD
    subgraph Familia
        P_Fam(["Predicción Familia"])
        T_Fam(["Etiqueta Real Familia"])
        CE_Fam["CrossEntropyLoss"]
        W_Fam{"x α"}
    end

    subgraph Género
        P_Gen(["Predicción Género"])
        T_Gen(["Etiqueta Real Género"])
        CE_Gen["CrossEntropyLoss"]
        W_Gen{"x β"}
    end

    subgraph Especie
        P_Esp(["Predicción Especie"])
        T_Esp(["Etiqueta Real Especie"])
        CE_Esp["CrossEntropyLoss"]
        W_Esp{"x γ"}
    end

    P_Fam --> CE_Fam
    T_Fam --> CE_Fam
    
    P_Gen --> CE_Gen
    T_Gen --> CE_Gen
    
    P_Esp --> CE_Esp
    T_Esp --> CE_Esp

    CE_Fam -.->|Pérdida Familia| W_Fam
    CE_Gen -.->|Pérdida Género| W_Gen
    CE_Esp -.->|Pérdida Especie| W_Esp

    SUM(("Suma\n(+)"))
    
    W_Fam ==> SUM
    W_Gen ==> SUM
    W_Esp ==> SUM

    TOTAL_LOSS{{"Pérdida Total a optimizar\nL = (α · Lf) + (β · Lg) + (γ · Ls)"}}
    
    SUM ===> TOTAL_LOSS

    classDef pred fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef loss fill:#ffebee,stroke:#c62828,stroke-width:2px;
    classDef weight fill:#fff8e1,stroke:#f57f17,stroke-width:2px;
    classDef sum fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef final fill:#4caf50,stroke:#1b5e20,stroke-width:3px,color:white;

    class P_Fam,T_Fam,P_Gen,T_Gen,P_Esp,T_Esp pred;
    class CE_Fam,CE_Gen,CE_Esp loss;
    class W_Fam,W_Gen,W_Esp weight;
    class SUM sum;
    class TOTAL_LOSS final;
```
