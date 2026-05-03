"""
Synthetic NIT-style Dataset Generator
=======================================
The original NIT dataset (Vu et al., Data in Brief 2025) is not yet publicly
hosted. This script generates a synthetic equivalent: 1000+ examples of
(natural language intent -> Juniper EX3300 configuration) pairs.

Why synthetic data is valid here:
  - The original NIT dataset is only 1000 examples
  - Juniper config syntax is deterministic and rule-based
  - We cover the same intent categories: interfaces, VLANs, BGP, ACLs, OSPF, NTP
  - We add controlled variation so the model learns patterns, not memorises strings

The output format exactly matches what the real NIT dataset uses:
  {"domain": "networking", "task": "intent_to_config",
   "instruction": "...", "output": "...", "source": "synthetic-nit"}
"""

import json
import random
from pathlib import Path
from itertools import product

random.seed(42)  # reproducible generation
OUT_PATH = Path("data/raw/networking/nit_synthetic.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def ip(a, b, c, d): return f"{a}.{b}.{c}.{d}"
def rand_ip(): return ip(random.randint(10,192), random.randint(0,254), random.randint(0,254), random.randint(1,254))
def rand_mask(): return random.choice(["255.255.255.0", "255.255.0.0", "255.255.255.252"])
def rand_prefix(): return random.choice([24, 25, 26, 27, 28, 30])
def rand_asn(): return random.randint(64512, 65534)  # private AS range
def rand_vlan(): return random.randint(10, 4000)
def rand_iface(): return f"ge-0/0/{random.randint(0,47)}"
def rand_community(): return f"{random.randint(100,999)}:{random.randint(100,999)}"
def rand_desc(): return random.choice(["uplink", "server-port", "mgmt", "core-link", "peer-link", "access-port"])


# ── Intent categories ────────────────────────────────────────────────────────
# Each function returns a (instruction, config) tuple

def gen_interface_ip():
    iface = rand_iface()
    addr = rand_ip()
    prefix = rand_prefix()
    desc = rand_desc()
    intents = [
        f"Configure interface {iface} with IP address {addr}/{prefix} and description {desc}",
        f"Set {iface} to IP {addr} with /{prefix} prefix length, label it {desc}",
        f"Assign {addr}/{prefix} to {iface} and add description {desc}",
    ]
    config = f"""interfaces {{
    {iface} {{
        description {desc};
        unit 0 {{
            family inet {{
                address {addr}/{prefix};
            }}
        }}
    }}
}}"""
    return random.choice(intents), config


def gen_interface_disable():
    iface = rand_iface()
    intents = [
        f"Disable interface {iface}",
        f"Shut down {iface}",
        f"Administratively disable {iface}",
    ]
    config = f"""interfaces {{
    {iface} {{
        disable;
    }}
}}"""
    return random.choice(intents), config


def gen_vlan_access():
    iface = rand_iface()
    vlan = rand_vlan()
    desc = rand_desc()
    intents = [
        f"Configure {iface} as an access port in VLAN {vlan} with description {desc}",
        f"Set {iface} to access mode, VLAN {vlan}, description {desc}",
        f"Assign VLAN {vlan} to access interface {iface}, label it {desc}",
    ]
    config = f"""interfaces {{
    {iface} {{
        description {desc};
        unit 0 {{
            family ethernet-switching {{
                interface-mode access;
                vlan {{
                    members {vlan};
                }}
            }}
        }}
    }}
}}"""
    return random.choice(intents), config


def gen_vlan_trunk():
    iface = rand_iface()
    vlans = sorted(random.sample(range(10, 500), 3))
    vlan_str = " ".join(str(v) for v in vlans)
    intents = [
        f"Configure {iface} as a trunk port allowing VLANs {', '.join(str(v) for v in vlans)}",
        f"Set {iface} to trunk mode with VLAN members {vlan_str}",
        f"Make {iface} a trunk interface carrying VLANs {vlan_str}",
    ]
    config = f"""interfaces {{
    {iface} {{
        unit 0 {{
            family ethernet-switching {{
                interface-mode trunk;
                vlan {{
                    members [ {vlan_str} ];
                }}
            }}
        }}
    }}
}}"""
    return random.choice(intents), config


def gen_bgp_neighbor():
    neighbor_ip = rand_ip()
    remote_asn = rand_asn()
    local_asn = rand_asn()
    intents = [
        f"Configure BGP neighbor {neighbor_ip} in AS {remote_asn} under local AS {local_asn}",
        f"Add BGP peer {neighbor_ip} with remote-as {remote_asn}, local AS is {local_asn}",
        f"Set up eBGP session to {neighbor_ip} (AS{remote_asn}) from AS{local_asn}",
    ]
    config = f"""routing-options {{
    autonomous-system {local_asn};
}}
protocols {{
    bgp {{
        group ebgp-peers {{
            type external;
            neighbor {neighbor_ip} {{
                peer-as {remote_asn};
            }}
        }}
    }}
}}"""
    return random.choice(intents), config


def gen_bgp_community():
    neighbor_ip = rand_ip()
    community = rand_community()
    intents = [
        f"Set BGP community {community} on routes sent to neighbor {neighbor_ip}",
        f"Tag outbound routes to {neighbor_ip} with community {community}",
        f"Apply community {community} to all prefixes advertised to {neighbor_ip}",
    ]
    config = f"""policy-options {{
    policy-statement set-community {{
        then {{
            community add comm-{community.replace(':','-')};
        }}
    }}
    community comm-{community.replace(':','-')} members {community};
}}
protocols {{
    bgp {{
        group ebgp-peers {{
            neighbor {neighbor_ip} {{
                export set-community;
            }}
        }}
    }}
}}"""
    return random.choice(intents), config


def gen_ospf_area():
    area = f"0.0.0.{random.randint(0,10)}"
    iface = rand_iface()
    intents = [
        f"Enable OSPF on interface {iface} in area {area}",
        f"Add {iface} to OSPF area {area}",
        f"Configure OSPF area {area} with interface {iface}",
    ]
    config = f"""protocols {{
    ospf {{
        area {area} {{
            interface {iface}.0;
        }}
    }}
}}"""
    return random.choice(intents), config


def gen_static_route():
    dest = f"{rand_ip()}/{rand_prefix()}"
    nexthop = rand_ip()
    intents = [
        f"Add a static route to {dest} via next-hop {nexthop}",
        f"Configure static route for {dest} pointing to {nexthop}",
        f"Install static route: destination {dest}, next hop {nexthop}",
    ]
    config = f"""routing-options {{
    static {{
        route {dest} next-hop {nexthop};
    }}
}}"""
    return random.choice(intents), config


def gen_ntp():
    ntp_server = rand_ip()
    intents = [
        f"Configure NTP server {ntp_server}",
        f"Set the NTP time source to {ntp_server}",
        f"Add {ntp_server} as an NTP server",
    ]
    config = f"""system {{
    ntp {{
        server {ntp_server};
    }}
}}"""
    return random.choice(intents), config


def gen_acl_permit():
    src = rand_ip()
    dst = rand_ip()
    prefix = rand_prefix()
    intents = [
        f"Create a firewall filter to permit traffic from {src} to {dst}/{prefix}",
        f"Allow packets sourced from {src} destined to {dst}/{prefix}",
        f"Permit {src} to reach {dst}/{prefix} in a firewall term",
    ]
    config = f"""firewall {{
    family inet {{
        filter permit-rule {{
            term allow-traffic {{
                from {{
                    source-address {src}/32;
                    destination-address {dst}/{prefix};
                }}
                then accept;
            }}
            term default-deny {{
                then discard;
            }}
        }}
    }}
}}"""
    return random.choice(intents), config


def gen_loopback():
    addr = f"10.255.{random.randint(0,254)}.{random.randint(1,254)}"
    intents = [
        f"Configure loopback interface lo0 with address {addr}/32",
        f"Set lo0 IP to {addr}/32",
        f"Assign {addr}/32 to the loopback interface",
    ]
    config = f"""interfaces {{
    lo0 {{
        unit 0 {{
            family inet {{
                address {addr}/32;
            }}
        }}
    }}
}}"""
    return random.choice(intents), config


def gen_snmp():
    community = random.choice(["public", "monitoring", "readonly", "netops"])
    mgmt_ip = rand_ip()
    intents = [
        f"Configure SNMP community {community} with client {mgmt_ip}",
        f"Enable SNMP read-only community {community} for host {mgmt_ip}",
        f"Set up SNMP with community string {community} restricted to {mgmt_ip}",
    ]
    config = f"""snmp {{
    community {community} {{
        authorization read-only;
        clients {{
            {mgmt_ip}/32;
        }}
    }}
}}"""
    return random.choice(intents), config


# ── Generator map ────────────────────────────────────────────────────────────

GENERATORS = [
    gen_interface_ip,
    gen_interface_ip,      # weighted double — most common task
    gen_interface_disable,
    gen_vlan_access,
    gen_vlan_access,       # weighted double
    gen_vlan_trunk,
    gen_bgp_neighbor,
    gen_bgp_neighbor,      # weighted double
    gen_bgp_community,
    gen_ospf_area,
    gen_static_route,
    gen_static_route,      # weighted double
    gen_ntp,
    gen_acl_permit,
    gen_loopback,
    gen_snmp,
]


# ── Main ─────────────────────────────────────────────────────────────────────

def generate(n=1200):
    records = []
    seen = set()

    while len(records) < n:
        gen_fn = random.choice(GENERATORS)
        instruction, output = gen_fn()

        # Deduplicate on instruction text
        key = instruction.strip().lower()
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "domain": "networking",
            "task": "intent_to_config",
            "instruction": instruction,
            "output": output,
            "source": "synthetic-nit",
        })

    return records


if __name__ == "__main__":
    print("Generating synthetic NIT-style dataset...")
    records = generate(n=1200)

    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"  Generated {len(records)} examples -> {OUT_PATH}")

    # Show a sample so you can verify quality
    print("\n── Sample record ──────────────────────────────────")
    sample = random.choice(records)
    print(f"Intent:  {sample['instruction']}")
    print(f"Config:\n{sample['output']}")
