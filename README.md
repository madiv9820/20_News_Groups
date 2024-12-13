# 20 News Groups

This project aims to classify news articles into 20 different genres using machine learning techniques. The dataset consists of **1000 text files** for each of the 20 genres, providing a total of **20,000 text files**.

## Dataset

The dataset is based on the **20 News Group** dataset, which includes a variety of news topics, such as politics, sports, technology, and more. Each genre is represented by text files containing articles or news stories related to that particular topic.

### Genres:

The following table lists the 20 genres included in the dataset:

| **Genre**                            | **Description**                 |
|--------------------------------------|---------------------------------|
| `alt.atheism`                        | Atheism discussions             |
| `comp.graphics`                      | Computer graphics               |
| `comp.os.ms-windows.misc`            | Miscellaneous Windows topics    |
| `comp.sys.ibm.pc.hardware`           | IBM PC hardware topics          |
| `comp.sys.mac.hardware`              | Macintosh hardware topics       |
| `comp.windows.x`                     | X Window System topics          |
| `misc.forsale`                       | Miscellaneous sales listings    |
| `rec.autos`                          | Automotive topics               |
| `rec.motorcycles`                    | Motorcycle-related discussions  |
| `rec.sport.baseball`                 | Baseball discussions            |
| `rec.sport.hockey`                   | Hockey discussions              |
| `sci.crypt`                          | Cryptography topics             |
| `sci.electronics`                    | Electronics-related discussions |
| `sci.med`                            | Medical-related discussions     |
| `sci.space`                          | Space-related discussions       |
| `soc.religion.christian`             | Christian religious discussions |
| `talk.politics.guns`                 | Gun control politics            |
| `talk.politics.mideast`              | Middle East political topics    |
| `talk.politics.misc`                 | Miscellaneous political topics  |
| `talk.religion.misc`                 | Miscellaneous religious topics |

## Data Processing

The raw text data from the text files is processed and transformed into a structured format, typically using **pandas** or other Python libraries, to create a **DataFrame**. This enables efficient manipulation, exploration, and analysis of the data for machine learning tasks.

## Objective

The main objective of this project is to build a classification model that can automatically categorize news articles into one of the 20 genres based on their content.