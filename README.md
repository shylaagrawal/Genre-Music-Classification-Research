# 1. Abstract

Music genre classification systems often report strong performance when evaluated on the same datasets used for training, yet their ability to generalize to real-world, out-of-domain music remains limited. This work investigates the cross-domain behavior of a convolutional neural network (CNN) genre classifier trained on the GTZAN dataset when applied to the MTG-Jamendo dataset. By comparing predicted labels from the GTZAN-trained model against genre annotations in MTG-Jamendo, we analyze how a narrow, ten-genre benchmark model performs on a broader and more diverse music collection. The classifier achieved an overall accuracy of 46.6%, performing best on dominant genres such as classical (precision 84.5%, recall 82.7%) and struggling with less prevalent genres like disco, blues, and country (precision <3.2%). Analysis of over- and under-classification patterns revealed systematic biases: disco, rock, and country were frequently over-predicted, while pop and jazz were substantially under-predicted. These findings highlight the risks of over-reliance on canonical datasets and motivate future research into more robust, transferable approaches for audio classification.

# 2. Introduction

Music genre classification is a foundational task in music information retrieval (MIR) with applications in recommendation systems, music discovery, cataloging, and large-scale audio analysis. In supervised genre classification, models are trained to assign one or more predefined genre labels to an audio clip using labeled examples. Modern approaches rely on features extracted directly from audio signals, such as spectral representations, rhythmic patterns, and timbral descriptors, which are then processed by machine learning models ranging from traditional classifiers to deep neural networks.

Despite steady improvements in model architectures, progress in genre classification has been strongly shaped by the datasets used for training and evaluation. One of the most influential benchmarks is the GTZAN dataset, introduced in 2002, which contains 1,000 audio clips evenly distributed across ten genres. Due to its accessibility and simplicity, GTZAN has become a de facto standard in music training and has been used to evaluate a wide range of genre classification models [1].

However, GTZAN exhibits several well-documented limitations. The dataset is small, genre coverage is narrow, and prior analyses have identified issues such as duplicate tracks, artist overlap across splits, and labeling errors [2]. As a result, models trained exclusively on GTZAN may overfit to dataset-specific artifacts and fail to generalize beyond the benchmark. In real-world settings, music spans a much wider range of styles, production qualities, and hybrid genres, often with ambiguous or multi-label annotations.

This project addresses the problem of domain shift in music genre classification by evaluating how a GTZAN-trained classifier performs on the MTG-Jamendo dataset, a large-scale and contemporary music tagging corpus. The central research questions are: 
- (1) How well does a genre classifier trained on GTZAN generalize to MTG-Jamendo?
- (2) Which genres transfer most effectively, and which exhibit systematic confusion or bias?
- (3) What does this evaluation reveal about the limitations of single-dataset benchmarking in MIR research? The goal of this work is to provide evidence of cross-domain performance gaps and to motivate more robust evaluation practices for music classification systems.

# 3. Related Work

Music genre classification has been widely studied for over two decades, with early work focusing on hand-crafted audio features and traditional classifiers [1]. More recent approaches employ deep learning models, particularly convolutional neural networks operating on time–frequency representations such as mel-spectrograms.

A recurring concern in this literature is dataset bias and evaluation reliability. Sturm [2] demonstrated that many published GTZAN results are inflated due to data leakage, repetition, and confounding factors unrelated to genre. Subsequent studies have emphasized the need for cleaner datasets and cross-dataset evaluation to assess true generalization performance [3].

Several datasets beyond GTZAN have been introduced to address these limitations. The Free Music Archive (FMA) dataset provides hierarchical genre annotations over tens of thousands of tracks [4]. The MTG-Jamendo dataset, released in 2019, offers large-scale multi-label annotations covering genres, moods, and instruments, and is specifically designed for music tagging research [5]. Compared to GTZAN, MTG-Jamendo reflects a more realistic distribution of contemporary music and annotation ambiguity.

To improve generalization to unseen or out-of-domain classes, some work has explored zero-shot and semantic learning approaches. These methods map audio features to semantic embedding spaces derived from text models such as Word2Vec or GloVe, enabling prediction of labels not seen during training [6]. While promising, zero-shot learning remains relatively underexplored in the specific context of genre classification, particularly for evaluating transfer from legacy benchmarks like GTZAN to modern datasets.

This work differs from prior studies by focusing explicitly on the cross-domain behavior of a widely used GTZAN-trained classifier when evaluated on MTG-Jamendo. Rather than proposing a new model, this study emphasizes evaluation methodology and error analysis, highlighting how performance degrades under domain shift and which genres are most affected.

# 4. Methods

## 4.1 Data

Two datasets were used in this study. The training domain consists of the GTZAN dataset, which contains 1,000 audio clips of 30 seconds each, evenly split across ten genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock [1]. GTZAN provides single-label genre annotations and has historically been used for closed-set classification.

The test domain is the MTG-Jamendo dataset, a large-scale Creative Commons music collection designed for music tagging research [5]. It includes tens of thousands of tracks annotated with multiple tags spanning genres, moods, and instruments. For this study, only genre-related tags were considered. Because MTG-Jamendo supports multi-label annotations and includes genres not present in GTZAN, evaluation reflects a realistic and challenging out-of-domain setting.

Data access and preprocessing followed the official MTG-Jamendo repository guidelines, including audio loading and metadata parsing. Audio clips were formatted to match the input requirements of the evaluated model.

## 4.2 Model

A convolutional neural network (CNN)-based music genre classifier trained on the GTZAN dataset was evaluated in this study. The model was trained for ten-class single-label classification corresponding to the GTZAN genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock [1]. No additional fine-tuning or retraining was performed.

The model was accessed via the Hugging Face model hub and inference was conducted using the Hugging Face audio classification pipeline within a Google Colab environment. For each input audio clip, the model outputs class probabilities over the ten GTZAN genres. The top predicted label was used as the model’s classification output.

Inference was performed on audio clips from the MTG-Jamendo dataset to evaluate cross-domain generalization. Raw prediction scores were recorded and normalized. Predicted genre labels were compared against MTG-Jamendo genre annotations to assess performance under domain shift.

To handle large-scale inference and allow resumption if processing was interrupted, batch processing was implemented:

```python
batch_size = 16
for batch_files in track_batches:  # track_batches is a list of lists of audio paths
    predictions = pipe(batch_files)
    for track, pred in zip(batch_files, predictions):
        results[track] = pred[0]["label"]  # store top prediction
```
From the multi-label MTG-Jamendo dataset, only tracks corresponding to the ten GTZAN genres were evaluated, ensuring a consistent cross-domain comparison. This step was necessary because MTG-Jamendo contains hundreds of tags, many of which do not appear in GTZAN. Filtering ensures the evaluation focuses on a shared label space and prevents spurious errors from irrelevant genres.

Relevant tracks were selected dynamically using the following logic:

```python
relevant_tracks = {}
for track_id, track_data in tracks.items():
    for track_genre in track_data["genre"]:
        if normalize(track_genre) in genres:
            file_name = track_data["path"]
            relevant_tracks[file_name] = track_data
            break  # only need one matching genre
```
## 4.3 Getting Results

For tracks appearing multiple times with differing predictions, only the first occurrence was used to ensure each audio file contributed exactly once to evaluation. 

[Figure 1]
| metric                          | description                                                            | value    |
|---------------------------------|------------------------------------------------------------------------|----------|
| fraction_with_secondary_label   | fraction of tracks that have a second true genre                       | 0.154474 |
| primary_only_accuracy           | accuracy considering only the primary genre                            | 0.466387 |
| accuracy_with_secondary_allowed | accuracy if a prediction matches either the primary or secondary genre | 0.517795 |
| absolute_accuracy_gain          | extra accuracy gained by allowing secondary genre matches              | 0.051409 |
| secondary_rescue_rate           | fraction of wrong predictions rescued only by the secondary label      | 0.049926 |


First, since there were some tracks in the dataset with two “true genres,” an evaluation of whether the secondary predicted genre was informative was necessary. Only 15% of tracks have secondary genre labels, and including them improves classification accuracy by just 5%, indicating that secondary genre annotations do not meaningfully affect overall evaluation. Hence, keeping them added complexity without meaningful benefit [Figure 1].

# 5. Results

For each genre, we report precision, recall, true count, and predicted count to characterize model behavior. Precision measures the proportion of tracks predicted as a given genre that are correctly labeled, while recall measures the proportion of tracks belonging to a given genre that the model successfully identifies. True count refers to the number of tracks in the dataset that actually belong to each genre, and predicted count refers to the total number of tracks assigned to each genre by the model, regardless of correctness. Together, true and predicted counts reveal systematic over- and under-classification across genres, particularly for less prevalent classes.

[Figure 2]

|     **genre** | **true_count** | **predicted_count** |
|----------:|-----------:|----------------:|
| classical |       1384 |            1355 |
|       pop |       1182 |             548 |
|      rock |        688 |             887 |
|      jazz |        328 |              85 |
|    hiphop |        154 |             103 |
|     metal |        115 |             206 |
|     blues |         89 |             165 |
|   country |         42 |             189 |
|    reggae |         38 |             142 |
|     disco |         26 |             354 |


[Figure 3]

<img width="502" height="209" alt="image" src="https://github.com/user-attachments/assets/b9c0bb07-4e1f-44f6-a1e0-de3bf0b192c9" />


The overall classification accuracy of the model was **46.6%**, with moderate performance across all genres. Performance varied substantially by genre: classical was the most accurately predicted genre, with a precision of 84.5% and recall of 82.7%, closely matching its high prevalence in the dataset (1384 tracks). In contrast, genres such as disco, blues, and country were predicted with very low precision (<3.2%), suggesting that the model frequently misclassified tracks into other genres [Figure 3].

Some genres exhibited inconsistent behavior between precision and recall. For example, metal had a relatively low precision of 29.1% but a higher recall of 52.2%, indicating that while many predicted metal tracks were incorrect, the model was relatively successful at capturing actual metal tracks. Conversely, pop and jazz showed higher precision than recall, suggesting that the model was conservative in predicting these genres, often missing tracks that truly belonged to them. Overall, these results reveal that the model performs best on dominant, well-represented genres (e.g., classical), struggles with less represented or more ambiguous genres (e.g., disco, blues), and exhibits systematic over or under classification patterns depending on the genre.

[Figure 4]

<img width="481" height="386" alt="image" src="https://github.com/user-attachments/assets/7518839f-a7e1-4159-91ec-7dd247ea4bd4" />


The over/under-classification analysis reveals strong systematic bias in the model’s predictions. Genres such as disco, rock, and country are heavily over-classified, indicating that they function as catch-all labels when the model is uncertain, which explains their low precision. In contrast, pop and jazz are substantially under-classified, showing that the model is conservative in assigning these genres and frequently fails to recognize them even when they are present [Figure 4]. Overall, this pattern suggests that misclassifications are skewed toward a small set of dominant genres, highlighting domain-dependent weaknesses in genre discrimination.

[Figure 5]

<img width="724" height="647" alt="image" src="https://github.com/user-attachments/assets/26f63759-7b3f-41b0-9808-494e8d98b504" />


The confusion matrix confirmed that classical stands out as the most stable genre, with a strong diagonal value (~83%), indicating both high precision and recall and relatively little confusion with other genres [Figure 5]. In contrast, pop and rock dominate the error patterns: pop is frequently misclassified as rock, and many genres (blues, country, metal) are disproportionately predicted as rock, confirming its role as a catch-all genre. Jazz and hiphop show low diagonal values, with jazz often confused with pop and classical, and hiphop frequently misclassified as pop or disco.

# 6. Conclusion

This section will summarize the research questions, key findings, and implications of the cross-domain evaluation. It will discuss the relevance of these findings for MIR research and outline directions for future work, including alternative datasets, multi-label evaluation, and domain-robust or zero-shot learning approaches.

# 7. References

[1] Tzanetakis, G., & Cook, P. (2002). *Musical genre classification of audio signals*. IEEE Transactions on Speech and Audio Processing.

[2] Sturm, B. L. (2014). *The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use*. arXiv preprint arXiv:1306.1461.

[3] Flexer, A. (2009). *A closer look on artist filters for musical genre classification*. Proceedings of the International Society for Music Information Retrieval Conference (ISMIR).

[4] Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). *FMA: A dataset for music analysis*. Proceedings of ISMIR.

[5] Bogdanov, D., Won, M., Tovstogan, P., Porter, A., & Serra, X. (2019). *The MTG-Jamendo dataset for automatic music tagging*. arXiv preprint arXiv:1905.06554.

[6] Choi, K., Fazekas, G., & Sandler, M. (2019). *Zero-shot learning for audio-based music classification and tagging*. arXiv preprint arXiv:1902.00716.

