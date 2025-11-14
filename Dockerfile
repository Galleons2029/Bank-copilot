# langgraph dockerfile -c langgraph.json Dockerfile

FROM langchain/langgraph-api:3.12

# -- Configure pip to use Tsinghua mirror --
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# -- Adding local package . --
ADD . /deps/Bank-copilot
# -- End of local package . --

# -- Installing all local dependencies --
RUN export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple && \
    export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple && \
    for dep in /deps/*; do \
        echo "Installing $dep"; \
        if [ -d "$dep" ]; then \
            echo "Installing $dep"; \
            (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple); \
        fi; \
    done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"agent": "/deps/Bank-copilot/app/core/agent/graph/instructor_agent.py:agent"}'



# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

WORKDIR /deps/Bank-copilot