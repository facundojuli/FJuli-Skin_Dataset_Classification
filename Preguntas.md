
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?

Rsta: Es necesario porque las capas lineales (nn.Linear) de una MLP calculan una multiplicación de matrices clásica que requiere que el vector de entrada tenga siempre un número de características fijo y estático. Si las imágenes tuvieran tamaños distintos, al estirarlas (nn.Flatten) generarían vectores de longitudes diferentes, lo que rompería la estructura rígida de los pesos entrenados del modelo y daría un error de dimensiones en PyTorch. Además, un tamaño uniforme es obligatorio para agrupar las imágenes en lotes (batches) de memoria homogéneos dentro del DataLoader.

- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?

Rsta: Albumentations ofrece ventajas clave frente a torchvision gracias a su alto rendimiento y velocidad de procesamiento, ya que está optimizada en OpenCV y NumPy.

- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?

Rsta: A.Normalize() normaliza los valores de los píxeles de la imagen (centrándolos y escalándolos según una media y desviación estándar), lo que ayuda a que la red neuronal entrene de forma más estable y rápida, ya que todas las entradas quedan en rangos similares y se evitan problemas derivados de trabajar directamente con valores entre 0 y 255.

- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?

Rsta: Convertimos las imágenes con ToTensorV2() al final de la pipeline porque transforma la imagen desde un arreglo de NumPy al formato tensor que utiliza PyTorch, permitiendo que la red neuronal pueda procesarla; además, se coloca al final porque las transformaciones de Albumentations trabajan sobre imágenes en formato NumPy antes de convertirlas a tensores.

## 2. Arquitectura del Modelo
- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

Rsta: Se utiliza una red MLP porque es un modelo simple y fácil de implementar para introducir conceptos básicos de clasificación de imágenes; sin embargo, tiene la limitación de que aplana la imagen y pierde la información espacial entre píxeles, por lo que suele obtener un rendimiento inferior al de una CNN, que puede detectar patrones locales como bordes, formas y texturas.

- ¿Qué hace la capa `Flatten()` al principio de la red?

Rsta: La capa Flatten() convierte la imagen multidimensional (alto, ancho y canales) en un único vector unidimensional, permitiendo que los datos puedan ser procesados por las capas totalmente conectadas (Linear) de la red MLP.

- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

Rsta: Se utilizó la función de activación ReLU (Rectified Linear Unit), ya que es simple y eficiente para entrenar redes profundas. No se usan Sigmoid o Tanh porque pueden saturarse para valores grandes o pequeños, provocando el problema del desvanecimiento del gradiente y haciendo que el entrenamiento sea más lento y menos efectivo.

- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

Rsta: Deberíamos cambiar el parámetro input_size y, en consecuencia, el tamaño de la primera capa lineal de la MLP, para que coincida con la nueva cantidad de píxeles resultante después de aplicar Flatten() a la imagen de mayor tamaño.

## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?

Rsta: optimizer.zero_grad() reinicia los gradientes almacenados de todos los parámetros del modelo antes de calcular los nuevos en la iteración actual, evitando que se acumulen gradientes de pasos anteriores y asegurando que la actualización de pesos se base únicamente en el lote actual.

- ¿Por qué usamos `CrossEntropyLoss()` en este caso?

Rsta: Usamos CrossEntropyLoss() porque el problema es de clasificación multiclase, donde cada imagen pertenece a una única categoría; esta función compara las predicciones del modelo con la clase correcta y es adecuada para entrenar redes que generan un puntaje para cada clase.

- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?

Rsta: El tamaño de batch_size determina cuántas muestras se procesan antes de actualizar los pesos: batches más grandes suelen producir gradientes más estables y aprovechar mejor el hardware, pero consumen más memoria, mientras que batches más pequeños introducen más ruido en el entrenamiento, aunque pueden ayudar a generalizar mejor y requieren menos memoria.

- ¿Qué pasaría si no usamos `model.eval()` durante la validación?

Rsta:  Si no usamos model.eval() durante la validación, las capas que se comportan de manera diferente entre entrenamiento y evaluación, como Dropout o BatchNorm, seguirán funcionando en modo entrenamiento, lo que puede producir métricas de validación inconsistentes y poco representativas del rendimiento real del modelo.

## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Rsta: Suele indicar overfitting, es decir, que el modelo aprendió muy bien los ejemplos de entrenamiento pero no generaliza correctamente a datos nuevos, por lo que su rendimiento disminuye cuando se evalúa sobre el conjunto de validación.

- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

Rsta: Podrían ser precision, recall y F1-score, ya que permiten evaluar mejor el impacto de los falsos positivos y falsos negativos, además, en conjuntos de datos desbalanceados, la accuracy puede resultar engañosa aunque el modelo tenga un rendimiento pobre sobre algunas clases.

- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

Rsta: La matriz de confusión muestra qué clases específicas está confundiendo el modelo y cuántos errores comete en cada una, mientras que la accuracy solo indica el porcentaje total de aciertos; por ello, permite identificar patrones de error y evaluar el desempeño de cada clase individualmente.

- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

Rsta: En el reporte de clasificación, precision indica qué proporción de las predicciones positivas realizadas por el modelo fueron correctas, recall mide qué proporción de los casos reales positivos fue detectada correctamente, y f1-score es la media armónica entre precision y recall, proporcionando una medida equilibrada del desempeño del modelo.

## 5. TensorBoard y Logging
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?

Rsta: TensorBoard permite visualizar en tiempo real métricas como la pérdida y la accuracy durante el entrenamiento y la validación, facilitando la detección de problemas como overfitting o underfitting, además de comparar experimentos y analizar la evolución del modelo de forma más clara y organizada.

- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?

Rsta: add_scalar se utiliza para registrar valores numéricos a lo largo del entrenamiento (por ejemplo, loss o accuracy), add_image permite visualizar imágenes, muestras o resultados del modelo, y add_text sirve para guardar información textual como hiperparámetros, comentarios o descripciones de experimentos dentro de TensorBoard.

- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?

Rsta: Guardar visualmente las imágenes de validación en TensorBoard permite inspeccionar los datos que recibe el modelo, verificar que las transformaciones se aplican correctamente y detectar posibles problemas en las imágenes o en las predicciones que no serían evidentes observando únicamente métricas numéricas.

- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

Rsta: Se pueden comparar distintos experimentos en TensorBoard registrándolos en directorios diferentes y visualizando sus curvas de métricas (como loss y accuracy) en un mismo gráfico, lo que permite analizar fácilmente cuál configuración o conjunto de hiperparámetros obtuvo mejor rendimiento.

## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

Rsta:  Si quisiéramos aplicar este modelo a un dataset con 100 clases, habría que modificar la última capa lineal para que tenga 100 neuronas de salida en lugar de la cantidad actual, de modo que el modelo pueda generar una predicción para cada clase; la función CrossEntropyLoss() seguiría siendo adecuada para el problema.

- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

Rsta: Una CNN es más adecuada porque utiliza convoluciones que capturan patrones locales de la imagen, requiere menos parámetros gracias al uso compartido de pesos, es más robusta a traslaciones y suele generalizar mucho mejor que una MLP en tareas de clasificación de imágenes.

- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?

Rsta: Si entrenamos el modelo con muy pocas imágenes por clase, es probable que sufra overfitting, aprendiendo características específicas de los ejemplos de entrenamiento pero sin lograr generalizar correctamente a nuevas imágenes, lo que reduciría su rendimiento en validación y prueba.

- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

Rsta: Para adaptar el pipeline a imágenes en escala de grises, habría que trabajar con un único canal en lugar de tres, ajustando la normalización para usar un solo valor de media y desviación estándar y modificando el input_size de la red para que coincida con el nuevo tamaño de la imagen luego de aplicar Flatten().

## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

Rsta: La regularización es un conjunto de técnicas que buscan reducir el overfitting, evitando que el modelo se adapte demasiado a los datos de entrenamiento; ejemplos comunes son Dropout, L2 regularization (weight decay) y el aumento de datos (data augmentation), que ayudan a mejorar la capacidad de generalización del modelo.

- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?

Rsta: Dropout desactiva aleatoriamente una fracción de neuronas durante el entrenamiento para evitar que dependan demasiado unas de otras, mientras que la regularización L2 (weight decay) penaliza los pesos grandes agregando un término extra a la función de pérdida, incentivando modelos más simples y con mejor capacidad de generalización.

- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?

Rsta: BatchNorm (Batch Normalization) normaliza las activaciones de una capa utilizando la media y desviación estándar del batch actual, lo que ayuda a estabilizar y acelerar el entrenamiento al mantener distribuciones de datos más consistentes entre capas, permitiendo además usar tasas de aprendizaje más altas y reduciendo el riesgo de sobreajuste.

- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?

Rsta: BatchNorm acelera la convergencia porque mantiene las activaciones de cada capa en rangos más estables durante el entrenamiento, reduciendo cambios bruscos en la distribución de los datos internos y permitiendo utilizar tasas de aprendizaje más altas sin comprometer la estabilidad del modelo.

- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?

Rsta: Sí, BatchNorm puede actuar como un regularizador suave porque introduce cierta variabilidad al normalizar usando las estadísticas de cada batch, lo que agrega ruido durante el entrenamiento y ayuda a reducir el overfitting, aunque generalmente no reemplaza técnicas específicas como Dropout o L2.

- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?

Rsta: Se observará que la loss de entrenamiento continúa disminuyendo y la accuracy de entrenamiento aumenta, mientras que la loss de validación comienza a aumentar o deja de mejorar y la accuracy de validación se estanca o disminuye, generando una brecha cada vez mayor entre ambas curvas.

- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

Rsta: La regularización mejora la generalización porque evita que el modelo memorice los datos de entrenamiento y lo obliga a aprender patrones más robustos y representativos, reduciendo el overfitting y logrando un mejor desempeño sobre datos nuevos no vistos durante el entrenamiento.

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?
Rsta: BatchNorm hizó que el entrenamiento sea más estable porque reduce las variaciones bruscas en las activaciones entre capas. Al normalizar las salidas de cada capa usando la media y la desviación estándar del batch, evita que los valores se vayan a rangos muy grandes o muy pequeños durante el entrenamiento.

- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?

Rsta: Sí, mejoró respecto a no tener nada.

- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?

Rsta: La combinación de BatchNorm y weight decay  fue la que produjo la mejor generalización.

- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?

Rsta: Sí, al usar BatchNorm se notaron cambios en la loss de entrenamiento, principalmente en su comportamiento más estable y en la velocidad de descenso respecto al nivel base.

## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?

Rsta: Es importante porque influye directamente en la estabilidad y velocidad del entrenamiento; una mala inicialización puede provocar gradientes muy pequeños o muy grandes, dificultando el aprendizaje o impidiendo que la red converja correctamente.

- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?

Rsta: Si todos los pesos se inicializan con el mismo valor, las neuronas de una misma capa aprenderán exactamente lo mismo durante el entrenamiento, produciendo el problema de simetría y evitando que la red aproveche su capacidad para aprender características diferentes, lo que perjudica gravemente el aprendizaje.

- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?

Rsta: La inicialización Xavier (Glorot) está diseñada para funciones de activación como Sigmoid o Tanh y busca mantener estable la varianza de las activaciones entre capas, mientras que la inicialización He está optimizada para ReLU y sus variantes, utilizando una varianza mayor para compensar que estas activaciones anulan los valores negativos y así evitar el desvanecimiento de los gradientes.

- ¿Por qué en una red con ReLU suele usarse la inicialización de He?

Rsta: En redes con ReLU suele usarse la inicialización de He porque está diseñada para compensar el hecho de que ReLU anula todas las activaciones negativas, manteniendo una varianza adecuada de las activaciones y los gradientes a través de las capas, lo que favorece un entrenamiento más estable y eficiente.

- ¿Qué capas de una red requieren inicialización explícita y cuáles no?

Rsta: Las capas que tienen parámetros aprendibles, como Linear (Dense) y Convolutional, requieren inicialización de pesos, mientras que capas sin parámetros entrenables, como ReLU, Flatten, Pooling o Dropout, no necesitan inicialización explícita porque no almacenan pesos que deban aprenderse.

### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?

Rsta: La de He fue la más rápida.
- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?

Rsta: La inicialización aleatoria

- ¿Qué impacto tiene la inicialización sobre las métricas de validación?

Rsta: EL modelo tiene el peor rendimiento cuando es inicializado aleatoriamente, pero la precisión fue menor a lo visto sin inicializar, dado que PyTorch hace una inicialización implícita que está mejor calibrada que la utilizada.

- ¿Por qué `bias` se suele inicializar en cero?

Rsta: Porque es una forma de inicio simple y clara, y no tiene el problema de la simetría porque los pesos están inicializados en valores distintos a 0.