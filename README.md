### Abstract:

Music genre classification models will often achieve strong results on familiar datasets, but struggle to __ to music from different domains. This paper evaluates a pre-trained GTZAN genre classification model when tested on the MTG-Jamendo dataset. By comparing true versus predicted genre labels, we analyze how well a model trained in a narrow, ten-genre dataset performs on broader, real-world data.  Results show __________________. These findings highlight the importance of evaluating music classification systems beyond their original datasets and motivate further exploration into _______________ approaches for audio classification.


### Intro:

Smth about what music genre classification is to the world. Models are usually trained to assign a predefined genre label to an audio clip, relying on supervised learning from curated datasets such as GTZAN. These models use featured extracted directly from the audio file, such as spectral patterns, rhythm and timbre, to predict labels like jazz, rock, or hip-hop.

However, many popular datasets are limited in size, diversity, and consistency. The GTZAN dataset, for example, contains only 1000 clips across ten genres, with overlapping samples and potential mislabels. As a result, models trained only on GTZAN may not be a fit for wider data, where it may encounter unknowns, performing well on in-scope music, but failing to accurately classify out of domain music.

This project aims to investigate how a model trained on GTZAN behaves when faced with out-of-domain data. Specifically, it explores the question: How well does a  GTZAN-trained genre classifier perform on the MTG-Jamendo dataset, and what patterns of errors or biases emerge? MTG-Jamendo provides a much broader and more contemporary range of music, making it an effective test to study for domain transfer, and data bias in genre recognition.

### Related Work:

Past papers mentioned here (add more)
Previous studies have proposed Zero-Shot Learning (ZSL) approaches to overcome these issues, mapping audio features to semantic embeddings (e.g. GloVe or Word2Vec) to predict unseen labels during training. While very promising, these methods remain underexplored in the context of genre classification as of now (check if this is true).

The GTZAN dataset, introduced in 2002, remains one of the most widely used benchmarks, containing 10 genres (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock). More recent datasets, such as the MTG-Jamendo collection introduced in 2019, include thousands of Creative Commons tracks annotated with multiple tags, spanning genres, moods, and instruments. 

MTG-Jamendo is 
Smth smth about why it may be better

How this work differs:
Talk about cross-domain behavior.

### Methods:

We used the MTG-Jamendo dataset, a large-scale Creative Commons collection designed for music tagging research. It includes tens of thousands of tracks with multi-label annotations (genres, moods, instruments, etc.) and diverse production styles, and only the genre tags were used for this analysis.

Test Data: MTG-Jamendo subset containing clips annotated with genre tags.
Training Data: GTZAN dataset with 10 classes: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock.

Preprocessing and dataset access was followed by the official https://github.com/MTG/mtg-jamendo-dataset/tree/master?tab=readme-ov-file#loading-data-in-python. 

The model evaluated was the GTZAN music genre classifier available on Hugging Face. This is a CNN-based model trained on GTZAN’s 10-genre classification. The model was evaluated by using it for inference on MTG-Jamendo audio files within a Google Colab notebook. Predicted labels were compared against MTG-Jamendos’s genre annotation to measure _________________________.


### Sources
https://arxiv.org/pdf/1306.1461
https://arxiv.org/html/2405.15096v1 


Stress on:

Despite lots of available datasets for classification: GTZAN is known as the “standard” dataset 
Mention some of the shortcomings of it
Smaller genres, smaller # of tracks
hasnt been a lot of 
Widely used -> cite other models that use it
How many times its been cited -> 
