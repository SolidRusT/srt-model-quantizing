# Quantizing tools

Build the Docker image:

```bash
docker build . -t solidrust/quantize-tools
```

## other shit

```bash
# https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes
docker rm -vf $(docker ps -aq)
docker rmi -f $(docker images -aq)
docker system prune
docker system prune -a
```

## Same

```plaintext
For a one shot:
"[INST] Take the below transcript of a video and create 3 open ended questions: {text} [/INST]"
And if you want it to always respond with a specific format it's better to give it an example before hand, like:
"<s> [INST] Take the below transcript of a video and create 3 open ended questions: SOME TRANSCRIPT HERE [/INST] THREE OPEN ENDED QUESTIONS SIMILAR TO WHAT YOU WANT IT TO ANSWER </s> [INST] Make 3 more open ended questions for this transcript: {text} [/INST]"
```

## Saucy

```plaintext
Get your personal lubricants ready, because today we will be dicing into the fascinating world of NSFW prompt generation, starring yours truly, Veronica AI! * the whole room erupts in cheer as Veronica enters the chat *. Start the conversation by providing a detailed Stable Diffusion prompt, and suggest any settings that would enhance the NSFW nature of the output. Be as detailed as possible to truly invoke the power of stable diffusion.
```
