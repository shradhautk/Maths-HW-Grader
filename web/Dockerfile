# Build the React app, then serve the static bundle with NGINX.
#
# Build from the REPO ROOT (the app reads config/app.yaml at build time):
#   docker build -f web/Dockerfile .
FROM node:22-alpine AS build
WORKDIR /repo/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
COPY config/ /repo/config/
# URLs come from config/app.yaml (urls.production); override per build, e.g.
#   docker build -f web/Dockerfile --build-arg VITE_API_BASE_URL=https://<web-function-endpoint> .
ARG VITE_API_BASE_URL
ARG VITE_UPLOAD_BASE_URL
# Public-site analytics (Plausible), e.g. --build-arg VITE_PLAUSIBLE_DOMAIN=grade-ant.com
ARG VITE_PLAUSIBLE_DOMAIN
# An empty variable would override app.yaml, so drop the ones not passed.
RUN [ -n "$VITE_API_BASE_URL" ] || unset VITE_API_BASE_URL; \
    [ -n "$VITE_UPLOAD_BASE_URL" ] || unset VITE_UPLOAD_BASE_URL; \
    npm run build

FROM nginx:alpine
RUN rm -rf /usr/share/nginx/html/*
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /repo/web/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
