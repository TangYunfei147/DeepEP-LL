"""CPU-only shared deployment contract for probe launchers."""
import ipaddress
import re
from pathlib import Path
from paths import project_path, REPO_ROOT

TESTS = ('empty', 'tiling', 'comm', 'metadata', 'native', 'network')

def load_config(path):
    import yaml
    class UniqueLoader(yaml.SafeLoader):
        pass
    def unique_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError('Duplicate YAML key: ' + str(key))
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    value = yaml.load(project_path(path).read_text(), Loader=UniqueLoader)
    if isinstance(value, list):
        value = {'ips': value}
    if not isinstance(value, dict):
        raise ValueError('Deployment YAML must be a mapping or legacy IPv4 list')
    allowed = {'ips', 'rank_num_per_server', 'port', 'shmem_port', 'timeout', 'run_id', 'output_root'}
    if set(value) - allowed:
        raise ValueError('Unknown deployment fields: ' + ', '.join(sorted(set(value)-allowed)))
    result = dict(rank_num_per_server=16, port=29500, shmem_port=29501, timeout=600, output_root='results')
    result.update(value)
    result['_source'] = str(project_path(path))
    return result

def make_plan(config, node_index=0, devices_per_host=None, test='comm', run_id='plan-only', validation=False):
    ips = config.get('ips')
    if not isinstance(ips, list) or not ips or any(not isinstance(ip, str) for ip in ips):
        raise ValueError('ips must be a nonempty IPv4 list')
    for ip in ips:
        address = ipaddress.ip_address(ip)
        if address.version != 4 or address.is_unspecified or address.is_multicast:
            raise ValueError('Use unicast bootstrap IPv4 addresses')
    if len(set(ips)) != len(ips):
        raise ValueError('Duplicate hosts')
    n = config.get('rank_num_per_server',16) if devices_per_host is None else devices_per_host
    if type(n) is not int or not 1 <= n <= 16:
        raise ValueError('rank_num_per_server must be an integer in 1..16')
    world = len(ips)*n
    if world not in (2,4,8,16,32) or (world == 32 and len(ips)<2):
        raise ValueError('Supported actual worlds: 2/4/8/16/32; EP32 requires multiple hosts')
    if type(node_index) is not int or not 0 <= node_index < len(ips):
        raise ValueError('Invalid node index')
    if test not in TESTS:
        raise ValueError('Unknown test')
    if validation and (len(ips)!=1 or n!=2):
        raise ValueError('Validation requires one host and exactly devices 0,1')
    port, shmem = config.get('port',29500), config.get('shmem_port',29501)
    if any(type(p) is not int or not 1024 <= p <=65535 for p in (port,shmem)) or port==shmem:
        raise ValueError('MASTER and SHMEM require distinct ports in 1024..65535')
    timeout = config.get('timeout',600)
    if type(timeout) is not int or not 10 <= timeout <=3600:
        raise ValueError('timeout must be 10..3600 seconds')
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',run_id):
        raise ValueError('Invalid run-id')
    output_root=project_path(config.get('output_root','results'))
    if output_root==REPO_ROOT or not output_root.is_relative_to(REPO_ROOT):
        raise ValueError('output_root must be inside the project')
    output=(output_root/run_id/f'node-{node_index}').resolve()
    if output==REPO_ROOT or not output.is_relative_to(REPO_ROOT):
        raise ValueError('Final output must remain inside the project')
    status='unsupported' if test=='native' and world<16 else 'supported'
    return dict(ips=ips, hosts=ips, local_world_size=n, devices_per_host=n, world=world,
                node_index=node_index, ranks=list(range(node_index*n,(node_index+1)*n)),
                master_addr=ips[0],master_port=port,shmem_port=shmem,endpoint=f'tcp://{ips[0]}:{shmem}',
                timeout=timeout,test=test,run_id=run_id,status=status,
                reason='Native LL requires EP16/32 in this test matrix' if status=='unsupported' else None,
                output=str(output),
                topology_source=config.get('_source','inline'),validation=validation)
