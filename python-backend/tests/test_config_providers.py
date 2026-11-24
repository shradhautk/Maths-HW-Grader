"""config/app.yaml deployment switch on the Python side (same rules as the Node backend)."""
import copy
import os
import sys

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [ROOT]

from gradeant_common import config  # noqa: E402

SHIPPED = os.path.join(ROOT, '..', 'config', 'app.yaml')


def shipped():
    with open(SHIPPED, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


@pytest.fixture()
def use_doc(monkeypatch, tmp_path):
    """Point the loader at an edited copy of the shipped file."""
    def write(doc, mode='production'):
        path = tmp_path / 'app.yaml'
        path.write_text(yaml.safe_dump(doc), encoding='utf-8')
        monkeypatch.setenv('APP_CONFIG_FILE', str(path))
        monkeypatch.setenv('APP_ENV', mode)
        monkeypatch.delenv('DATABASE_PROVIDER', raising=False)
        config.app_config.cache_clear()
    yield write
    config.app_config.cache_clear()


def test_aws_target_is_aws(use_doc):
    use_doc({**shipped(), 'deployment': 'aws'})
    assert config.providers() == config.AWS_PROVIDERS
    assert config.database() == 'dynamodb'


def test_shipped_file_selects_hybrid(use_doc):
    use_doc(shipped())
    assert config.providers()['database'] == 'supabase'


def test_missing_section_means_aws(use_doc):
    doc = shipped()
    doc.pop('deployment')
    doc.pop('deployments')
    use_doc(doc)
    assert config.database() == 'dynamodb'


def test_hybrid_selects_supabase_in_production_only(use_doc):
    doc = shipped()
    doc['deployment'] = 'hybrid'
    use_doc(doc)
    assert config.database() == 'supabase'
    assert config.providers()['grading'] == 'sqs'
    use_doc(doc, mode='development')
    assert config.database() == 'dynamodb'


def test_env_override_wins(use_doc, monkeypatch):
    use_doc(shipped())
    monkeypatch.setenv('DATABASE_PROVIDER', 'supabase')
    assert config.database() == 'supabase'
    monkeypatch.setenv('DATABASE_PROVIDER', 'mongodb')
    with pytest.raises(RuntimeError, match='DATABASE_PROVIDER must be one of dynamodb, supabase'):
        config.database()


@pytest.mark.parametrize('mutate, message', [
    (lambda d: d.__setitem__('deployment', 'heroku'), 'deployment: "heroku" is not defined'),
    (lambda d: d['deployments']['hybrid']['providers'].__setitem__('database', 'mongodb'),
     'providers.database must be one of dynamodb, supabase'),
    (lambda d: d['deployments']['hybrid']['providers'].__setitem__('queue', 'kafka'), 'providers.queue: unknown provider'),
])
def test_mistakes_stop_startup(use_doc, mutate, message):
    doc = copy.deepcopy(shipped())
    mutate(doc)
    use_doc(doc)
    with pytest.raises(RuntimeError, match=message.replace('.', r'\.').replace('(', r'\(')):
        config.app_config()


def test_node_and_python_accept_the_same_values():
    """The two copies of the provider list must not drift."""
    node = open(os.path.join(ROOT, '..', 'backend', 'src', 'config', 'appConfig.js'), encoding='utf-8').read()
    for provider, values in config.PROVIDERS.items():
        assert f"{provider}: [{', '.join(repr(v) for v in values)}]" in node, provider
