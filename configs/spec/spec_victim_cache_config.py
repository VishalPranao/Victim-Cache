

import argparse
import sys
import os

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.params import NULL
from m5.util import addToPath, fatal, warn

addToPath('../')

from ruby import Ruby
from common import Options
from common import Simulation
from common import CacheConfig
from common import CpuConfig
from common import ObjectList
from common import MemConfig
from common.FileSystemConfig import config_filesystem
from common.Caches import *
from common.cpu2000 import *

import spec2k6_spec2k17


# L1 Cache: 16KB direct-mapped, 2 cycle, writeback_clean=True
class VictimL1DCache(Cache):
    # Size and associativity
    size = '16kB'
    assoc = 1  
    
    # Timing parameters
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    
    mshrs = 4             
    tgts_per_mshr = 16     
    write_buffers = 8
    
    clusivity = 'mostly_incl'
    
    tags = BaseSetAssoc()
    replacement_policy = LRURP() 
    
    # Disable prefetching
    prefetcher = NULL

class VictimL1ICache(VictimL1DCache):
    is_read_only = True

    writeback_clean = False

# L2 Cache (Simulating Victim Cache): 64 entries, fully-associative, 1 cycle
class VictimCacheL2(Cache):
    # Size: 64 entries Ã— 64B blocks = 4KB
    size = '4kB'  # 64 entries
    
    assoc = 64 
    
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    
    mshrs = 8
    tgts_per_mshr = 12
    write_buffers = 4
    
    clusivity = 'mostly_excl'

    tags = FALRU()
    
    # Disable prefetching
    prefetcher = NULL
    victim_only = True

# L3 Cache (Real L2): 128KB, 4-way, 20 cycle
class RealL2CacheL3(Cache):
    size = '128kB'
    assoc = 4  # 4-way set-associative
    
    tag_latency = 20
    data_latency = 20
    response_latency = 20
    
    mshrs = 20           
    tgts_per_mshr = 12   
    write_buffers = 8    
    
    clusivity = 'mostly_incl'
    
    writeback_clean = False
    
    tags = BaseSetAssoc()
    replacement_policy = LRURP()
    
    # Disable prefetching
    prefetcher = NULL

def get_processes(args):
    """Interprets provided args and returns a list of processes"""
    multiprocesses = []
    inputs = []
    outputs = []
    errouts = []
    pargs = []

    workloads = args.cmd.split(';')
    if args.input != "":
        inputs = args.input.split(';')
    if args.output != "":
        outputs = args.output.split(';')
    if args.errout != "":
        errouts = args.errout.split(';')
    if args.options != "":
        pargs = args.options.split(';')

    idx = 0
    for wrkld in workloads:
        process = Process(pid = 100 + idx)
        process.executable = wrkld
        process.cwd = os.getcwd()
        process.gid = os.getgid()

        if args.env:
            with open(args.env, 'r') as f:
                process.env = [line.rstrip() for line in f]

        if len(pargs) > idx:
            process.cmd = [wrkld] + pargs[idx].split()
        else:
            process.cmd = [wrkld]

        if len(inputs) > idx:
            process.input = inputs[idx]
        if len(outputs) > idx:
            process.output = outputs[idx]
        if len(errouts) > idx:
            process.errout = errouts[idx]

        multiprocesses.append(process)
        idx += 1

    if args.smt:
        assert(args.cpu_type == "DerivO3CPU")
        return multiprocesses, idx
    else:
        return multiprocesses, 1

parser = argparse.ArgumentParser(description='Gem5 SPEC config with victim cache simulation')
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

if '--ruby' in sys.argv:
    Ruby.define_options(parser)

parser.add_argument("-b", "--benchmark", default="",
                 help="The benchmark to be loaded.")

# Victim Cache Control
parser.add_argument("--enable-victim-cache", action='store_true', dest='victim_cache_enabled',
                    help="Enable victim cache (3-level: L1â†’L2(victim)â†’L3)")
parser.add_argument("--disable-victim-cache", action='store_false', dest='victim_cache_enabled',
                    help="Disable victim cache (2-level: L1â†’L2, L2 is 128KB like baseline)")
parser.set_defaults(victim_cache_enabled=True)  # Default: victim cache ON

# Fast-forward and Simulation Control Parameters
parser.add_argument("--ff-instructions", type=int, default=0,
                    help="Number of instructions to fast-forward (default: 0)")
parser.add_argument("--sim-instructions", type=int, default=0,
                    help="Number of instructions to simulate after fast-forward (default: 0, no limit)")
parser.add_argument("--enable-fast-forward", action='store_true',
                    help="Enable fast-forward: 1B fast-forward + 100M simulation")
parser.add_argument("--measure-ff-time", action='store_true',
                    help="Measure and report fast-forward execution time")

args = parser.parse_args()


if args.enable_fast_forward:
    args.ff_instructions  = 1000000000  # 1B instructions
    args.sim_instructions = 100000000    # 100M instructions
    print(f"Fast-forward mode: {args.ff_instructions:,} FF, {args.sim_instructions:,} detailed")
elif args.ff_instructions > 0:
    print(f"Custom fast-forward: {args.ff_instructions:,} FF, {args.sim_instructions:,} detailed")

if args.ff_instructions > 0:
    args.fast_forward = str(args.ff_instructions)
    if args.sim_instructions > 0:
        args.maxinsts = args.sim_instructions
        print(f"Will fast-forward {args.ff_instructions:,} then simulate {args.sim_instructions:,} instructions")
    else:
        args.maxinsts = 1
        print(f"Fast-forward only: {args.ff_instructions:,} instructions")
elif args.sim_instructions > 0:
    args.maxinsts = args.sim_instructions
    print(f"Pure detailed simulation: {args.sim_instructions:,} instructions")

# Library redirects
args.redirects = ['/lib64=/package/gcc/8.3.0/lib64']

multiprocesses = []
numThreads = 1

process = spec2k6_spec2k17.get_process(args, buildEnv['TARGET_ISA'])
multiprocesses.append(process)

(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(args)
CPUClass.numThreads = numThreads


if args.smt and args.num_cpus > 1:
    fatal("Cannot use SMT with multiple CPUs!")

np = args.num_cpus
mp0_path = multiprocesses[0].executable

system = System(cpu = [CPUClass(cpu_id=i) for i in range(np)],
                mem_mode = test_mem_mode,
                mem_ranges = [AddrRange(args.mem_size)],
                cache_line_size = args.cacheline_size)

if numThreads > 1:
    system.multi_thread = True

# Clock domains
system.voltage_domain = VoltageDomain(voltage = args.sys_voltage)
system.clk_domain = SrcClockDomain(clock = args.sys_clock,
                                   voltage_domain = system.voltage_domain)
system.cpu_voltage_domain = VoltageDomain()
system.cpu_clk_domain = SrcClockDomain(clock = args.cpu_clock,
                                       voltage_domain = system.cpu_voltage_domain)

if args.elastic_trace_en:
    CpuConfig.config_etrace(CPUClass, system.cpu, args)

for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

if ObjectList.is_kvm_cpu(CPUClass) or ObjectList.is_kvm_cpu(FutureClass):
    if buildEnv['TARGET_ISA'] == 'x86':
        system.kvm_vm = KvmVM()
        system.m5ops_base = 0xffff0000
        for process in multiprocesses:
            process.useArchPT = True
            process.kvmInSE = True
    else:
        fatal("KvmCPU can only be used in SE mode with x86")

if args.simpoint_profile:
    if not ObjectList.is_noncaching_cpu(CPUClass):
        fatal("SimPoint/BPProbe should be done with an atomic cpu")
    if np > 1:
        fatal("SimPoint generation not supported with more than one CPUs")

for i in range(np):
    if args.smt:
        system.cpu[i].workload = multiprocesses
    elif len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]

    if args.simpoint_profile:
        system.cpu[i].addSimPointProbe(args.simpoint_interval)

    if args.checker:
        system.cpu[i].addCheckerCpu()

    if args.bp_type:
        bpClass = ObjectList.bp_list.get(args.bp_type)
        system.cpu[i].branchPred = bpClass()

    if args.indirect_bp_type:
        indirectBPClass = ObjectList.indirect_bp_list.get(args.indirect_bp_type)
        system.cpu[i].branchPred.indirectBranchPred = indirectBPClass()

    system.cpu[i].createThreads()


if args.ruby:
    Ruby.create_system(args, False, system)
    assert(args.num_cpus == len(system.ruby._cpu_ports))
    system.ruby.clk_domain = SrcClockDomain(clock = args.ruby_clock,
                                        voltage_domain = system.voltage_domain)
    for i in range(np):
        ruby_port = system.ruby._cpu_ports[i]
        system.cpu[i].createInterruptController()
        ruby_port.connectCpuPorts(system.cpu[i])
        
else:
    # Classic memory system (non-Ruby)
    MemClass = Simulation.setMemClass(args)
    system.membus = SystemXBar()
    system.system_port = system.membus.cpu_side_ports
    
    if args.victim_cache_enabled:
        system.l3 = RealL2CacheL3(clk_domain=system.cpu_clk_domain)
        print("L3 (Real L2): 128KB, 4-way, 20 cycles, mostly_incl")
        
        # L2 Cache (Victim Cache): 64 entries, fully-assoc, 1 cycle, mostly_excl
        # This simulates a victim cache between L1 and L3
        system.l2 = VictimCacheL2(clk_domain=system.cpu_clk_domain)
        print("L2 (Victim):  4KB (64 entries), fully-assoc, 1 cycle, mostly_excl")
        
        # Create crossbars for three-level hierarchy
        # L1 â† tol2bus â†’ L2 â† tol3bus â†’ L3 â† membus â†’ Memory
        system.tol2bus = L2XBar(clk_domain=system.cpu_clk_domain,
                                width=32)  # 32B width for L1-L2 connection
        system.tol3bus = L2XBar(clk_domain=system.cpu_clk_domain,
                                width=32)  # 32B width for L2-L3 connection
        
        # Connect L3 (real L2) to memory bus
        system.l3.cpu_side = system.tol3bus.mem_side_ports
        system.l3.mem_side = system.membus.cpu_side_ports
        
        # Connect L2 (victim cache) between tol2bus and tol3bus
        system.l2.cpu_side = system.tol2bus.mem_side_ports
        system.l2.mem_side = system.tol3bus.cpu_side_ports
        
    else:
        system.l2 = RealL2CacheL3(clk_domain=system.cpu_clk_domain)
        print("L2: 128KB, 4-way, 20 cycles, mostly_incl")
        
        # Create single crossbar for two-level hierarchy
        system.tol2bus = L2XBar(clk_domain=system.cpu_clk_domain,
                                width=32)
        
        # Connect L2 to memory bus
        system.l2.cpu_side = system.tol2bus.mem_side_ports
        system.l2.mem_side = system.membus.cpu_side_ports
        
    
    for i in range(np):
        # Create interrupt controller (required for X86)
        system.cpu[i].createInterruptController()
        
        icache = VictimL1ICache()
        dcache = VictimL1DCache()
        
        # Set writeback_clean based on configuration
        if args.victim_cache_enabled:
            # Victim cache config: Send L1 evictions to victim cache
            dcache.writeback_clean = True
            icache.writeback_clean = False 
        else:
            # Baseline config: Normal L1 behavior (no writeback of clean lines)
            dcache.writeback_clean = False
            icache.writeback_clean = False
        
        # Page table walker caches (if needed)
        if buildEnv['TARGET_ISA'] in ['x86', 'riscv']:
            iwalkcache = PageTableWalkerCache()
            dwalkcache = PageTableWalkerCache()
        else:
            iwalkcache = None
            dwalkcache = None
        
        # Connect L1 caches to CPU
        system.cpu[i].addPrivateSplitL1Caches(icache, dcache,
                                              iwalkcache, dwalkcache)
        
        # Connect L1 to tol2bus (which connects to L2 victim cache)
        system.cpu[i].icache.mem_side = system.tol2bus.cpu_side_ports
        system.cpu[i].dcache.mem_side = system.tol2bus.cpu_side_ports
        
        if iwalkcache:
            iwalkcache.mem_side = system.tol2bus.cpu_side_ports
            dwalkcache.mem_side = system.tol2bus.cpu_side_ports
        
        # For X86, connect interrupt controller to memory
        if buildEnv['TARGET_ISA'] == 'x86':
            system.cpu[i].interrupts[0].pio = system.membus.mem_side_ports
            system.cpu[i].interrupts[0].int_requestor = system.membus.cpu_side_ports
            system.cpu[i].interrupts[0].int_responder = system.membus.mem_side_ports
    

    args.mem_latency = '150ns'
    
    MemConfig.config_mem(args, system)
    
    # Override memory controller latencies to achieve 300 CPU cycles
    # 300 cycles @ 2GHz (0.5ns/cycle) = 150ns
    # Split: 75ns frontend + 75ns backend = 150ns total
    if hasattr(system, 'mem_ctrls'):
        print("-"*80)
        print("Memory Configuration:")
        for mem_ctrl in system.mem_ctrls:
            if hasattr(mem_ctrl, 'static_frontend_latency'):
                mem_ctrl.static_frontend_latency = '75ns'
            if hasattr(mem_ctrl, 'static_backend_latency'):
                mem_ctrl.static_backend_latency = '75ns'
            if hasattr(mem_ctrl, 'dram'):
                if hasattr(mem_ctrl.dram, 'latency'):
                    mem_ctrl.dram.latency = '150ns'
        print(f"  Memory latency: 300 CPU cycles (150ns @ 2GHz)")
        print(f"  Frontend: 75ns, Backend: 75ns")
        print("="*80)
    
    config_filesystem(system, args)

system.workload = SEWorkload.init_compatible(mp0_path)

if args.wait_gdb:
    system.workload.wait_for_remote_gdb = True

root = Root(full_system = False, system = system)


if args.ff_instructions > 0 and args.measure_ff_time:
    import time
    print(f"\nStarting fast-forward: {args.ff_instructions:,} instructions...\n")
    start_time = time.time()
    
    exit_event = Simulation.run(args, root, system, FutureClass)
    
    end_time = time.time()
    ff_duration = end_time - start_time
    
    print(f"\nFast-forward completed!")
    print(f"Execution time: {ff_duration:.2f} seconds")
    print(f"Rate: {args.ff_instructions / ff_duration:,.0f} inst/s")
    if args.sim_instructions > 0:
        print(f"Total simulated instructions: {args.ff_instructions + args.sim_instructions:,}")
else:
    if args.ff_instructions > 0:
        print(f"\nExecuting with fast-forward: {args.ff_instructions:,} instructions")
        if args.sim_instructions > 0:
            print(f"Detailed simulation: {args.sim_instructions:,} instructions\n")
    
    Simulation.run(args, root, system, FutureClass)
