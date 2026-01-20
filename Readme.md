# **PGMAN**

---

## Preparation

**VideoLLaMA2-7B** model, refer to its official HuggingFace page for details:

- [VideoLLaMA2-7B @ HuggingFace](https://huggingface.co/VideoLLaMA/VideoLLaMA2-7b)

---

## Project Structure

The project root directory (specified via `--root_path`) is recommended to follow this structure:

```text
<root_path>/  
├── MeiTu/
│   ├── video/           
│   └── audio/           
├── annotations/         
│   ├── mtsvrc_title.json
│   └── mtsvrc_label.json
└── results/             
    └── main/            
```

## Running Instructions

You can train and evaluate the model quickly with the following command:

```bash
$ python main.py
```

## Datasets

The datasets used include: [Ekman-6](https://drive.google.com/drive/folders/0B-iork9xj4brQmlYYjlsUUtVVGM?resourcekey=0-ZCZMvJXTIPVYAIcXyGH7cA), [VideoEmotion-8](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA12).
The expanded ME-5 dataset will be released in the future.