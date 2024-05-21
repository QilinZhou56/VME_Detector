# Project Proposal

## Team Members:
Jaskirat, Grey, Qilin (CAPPies)

## Related repository:
Our primary repository is [here](https://github.com/jaskcodes/Reef-madness/tree/main/detection_opencv_streamlit/application). We use **this only to connect with streamlit** with trained weights files and image repositories. 

## Application Use （Local; preferred one as all tasks worked well)
```bash
streamlit run cv_models.py
```

## Application Use （Public; hosted by streamlit)
This is the associated [URL](https://vmedetector-7bsaxaqq9o7wrahnbnbqxk.streamlit.app/). **Only upload images and explore species map EDA in this case**. Due to 1GB limit in streamlit cloud, real-time classification or detection, zero-shot detection, and access to local image repositories may fail in the mid of execution. Whenever it cracks, click **reboot** to reactivate the app in **manage my app**.

## Directory 
├── app
│   ├── css
│   │   ├── **/*.css
│   ├── favicon.ico
│   ├── images
│   ├── index.html
│   ├── js
│   │   ├── **/*.js
│   └── partials/template
├── dist (or build)
├── node_modules
├── bower_components (if using bower)
├── test
├── Gruntfile.js/gulpfile.js
├── README.md
├── package.json
├── bower.json (if using bower)
└── .gitignore

## 1. Classification of Visually Similar Species

![schema](Marine%20Species%20Detection%20OpenCV/schema.png)

### Overview
This part of the project focuses on distinguishing between species that are visually similar using advanced computer vision techniques.

### Species Selection
- **Abyssopathes lyra** - A black coral similar to others in its genus. (2 images)
- **Abyssopathes** - Includes various species within the genus. (30 images)
- **Abyssopathes sp.** - Another member of the Abyssopathes genus. (13 images)
- **Anthomastus sp.** - Visually similar to other Anthomastus corals. (79 images)
- **Anthomastus tahinodus** - Similar to Anthomastus sp.
- **Paragorgia arborea (Bubblegum Coral)** - Often similar to other Paragorgia species.
- **Paragorgia sp.** - Similar to Paragorgia arborea.
- **Corallium sp.** - Deep-sea corals often similar across the genus.

### Data Collection
Images will be gathered from databases like FishBase or directly requested from marine research institutions.

### Model Development
We plan to utilize a CNN architecture, possibly starting with pretrained models such as ResNet or VGG, which will be fine-tuned on our fathom image dataset.

### Evaluation
The model's performance will be assessed using metrics like accuracy, precision, recall, and F1-score.

## 2. Identification of Endangered Species

### Overview
Focusing on endangered species, this project aims to detect and count individuals within images or video footage, aiding in conservation efforts.

### Species Selection
- **Antipatharia** - Vulnerable due to habitat needs and ocean acidification.
- **Bathypathes patula** - At risk from deep-sea fishing.
- **Corallium** - Harvested for jewelry, placing some species at risk.
- **Paragorgia arborea** - At risk from deep-sea trawling.
- **Primnoa** - Sensitive to temperature changes and disturbances.
- **Sebastes aleutianus-melanostictus complex** - Overfished and vulnerable.
- **Holothuria scabra** - Facing global overfishing pressures.
- **Pristipomoides filamentosus** - Vulnerable due to fishing pressures.
- **Lophiiformes (Monkfish)** - Threatened by commercial fishing.
- **Hippoglossus hippoglossus (Atlantic Halibut)** - Overfished and habitat disrupted.

### Data Collection
Sufficient images for each selected species will be gathered from the fathom dataset or deepfish.ai, including various contexts and environments.


