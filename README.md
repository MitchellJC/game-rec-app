## Activating Environment
```bash
source .venv/Scripts/activate
```

## Building Docker
Open docker desktop, run following in terminal.
```bash
docker compose -f docker-compose.dev.yml up --build
```

## Making Changes to Prod
Open bash terminal and run the following to sign in
```bash
heroku open -a dry-eyrie-18912 \
& heroku container:login
```
To push and release changes run.
```bash
heroku container:push web
heroku container:release web
```

## Related Repositories
### game-rec-app
Video game recommendation web-app.

### Data and Modelling
Data and modelling repository can be found [here](https://github.com/MitchellJC/game-rec).

### Image Collection
Image collection repository can be found [here](https://github.com/MitchellJC/game-rec-scrape).