# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import os
import random
import subprocess
import time

STATE_FREE = "free"  # resource is ready for use
STATE_BUSY = "busy"  # resource is not ready, in use
STATE_DONE = "done"  # single-use resource has been consumed
STATE_UNKNOWN = "unknown"  # resource is in unknown state, usually treated as an
                           # error
STATE_ERROR = "error" # resource is not ready, error occured
STATES = [ STATE_FREE, STATE_BUSY, STATE_DONE, STATE_UNKNOWN, STATE_ERROR ]

class Queue(object):
  """Maintains the state for each element in a fixed list of objects."""

  def __init__(self, states, objects):
    """Create new Queue object with given states. Initially, all objects are
    assigned to state[0]."""
    states = list(states)
    assert(len(states) > 0)
    self.states = states
    self.lists = {}
    self.lists[states[0]] = list(objects)
    for state in states[1:]:
      self.lists[state] = []
    self.objects = list(objects)

  def SetState(self, obj, state):
    assert(state in self.states)
    assert(obj in self.objects)
    for s in self.states:
      objects = self.lists[s]
      if obj in objects:
        objects.remove(obj)
    self.lists[state].append(obj)

  def AllInState(self, state):
    """Determine if all objects are in given state."""
    assert(state in self.states)
    return len(self.lists[state]) == len(self.objects)

  def AnyInState(self, state):
    """Determine if any objects are in given state."""
    assert(state in self.states)
    return len(self.lists[state]) > 0

  def InState(self, state):
    """Get list of objects in given state."""
    assert(state in self.states)
    return list(self.lists[state])

  def ChooseRandom(self, state):
    """Choose random object in given state."""
    assert(state in self.states)
    objects = self.lists[state]
    assert(len(objects) > 0)
    return random.choice(objects)

  def ChooseNext(self, state):
    """Choose first object in given state."""
    assert(state in self.states)
    objects = self.lists[state]
    assert(len(objects) > 0)
    return objects[0]

class Network(object):
  """Handles remote command invocations."""

  def __init__(self, results_dir, clusters, verbose = False):
    """Create new network object.
    results_dir -- Directory on remote machines in which to store results
    clusters -- (dictionary) mapping from a management node for a cluster to a
                list of worker nodes. The management node is used only to check
                job status, while worker nodes actually run those jobs.
    """
    self.results_dir = results_dir
    self.clusters = clusters
    self.verbose = verbose

  def GetHosts(self):
    hosts = []
    for member_hosts in self.clusters.values():
      hosts.extend(member_hosts)
    return hosts

  def _GetClusterForHost(self, host):
    for cluster, hosts in self.clusters.items():
      if host in hosts:
        return cluster
    assert(false), "Unknown host: %s" % host

  def _GetClusterStatus(self, cluster_host, jobs):
    """Get the status of all jobs running on the given cluster.
    cluster_host -- (string) maintenance node for cluster
    jobs -- (JobSpec list) specs for jobs of interest
    Requires that the command 'local-job-status' be on the remote path of the
    maintenance node, and that 'ssh' be on the local path.
    """
    cmds = [ "cd %s" % self.results_dir ]
    cmds += [ "gjob local status %s" % job.id_ for job in jobs ]
    cmd = "\n".join(cmds)
    args = [ "ssh", "-q", cluster_host ]
    p = subprocess.Popen(args, stdin = subprocess.PIPE,
        stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    (response, stderr) = p.communicate(cmd)
    if not (stderr == None or stderr == ""):
      raise Exception("Job status update command failed on cluster host "
          "%s: (stderr below)\n%s" % (cluster_host, stderr))
    results = []
    for job, line in zip(jobs, response.split("\n")):
      args = line.strip().split()
      if len(args) != 2:
        raise Exception("Error parsing server response. Response is below.\n"
            "%s" % response)
      returned_job_id, returned_state = args
      if returned_job_id != job.id_:
        raise Exception("Error parsing server response -- results in wrong "
            "order. Response is below.\n%s" % response)
      if returned_state not in STATES:
        raise Exception("Error parsing server response -- "
            "invalid state (%s) for job (%s). Response is below.\n%s" % \
            (returned_state, returned_job_id, response))
      results.append(returned_state)
    if len(results) != len(jobs):
      raise Exception("Error parsing server response -- missing status for "
          "some jobs. Response is below.%s\n" % response)
    return results

  def GetJobStatus(self, jobs):
    """Get the current state for a list of jobs."""
    # Group jobs by cluster
    clusters = {}
    for job in jobs:
      # Get cluster for job host
      job_cluster = None
      for cluster, hosts in self.clusters.items():
        if job.host in hosts:
          job_cluster = cluster
      if job_cluster == None:
        raise Exception("Unknown job host: %s" % job.host)
      if job_cluster not in clusters:
        clusters[job_cluster] = [ job ]
      else:
        clusters[job_cluster].append(job)
    # Make single job-status request per cluster
    results = []
    for cluster, jobs in clusters.items():
      results.extend(self._GetClusterStatus(cluster, jobs))
    return results

  def GetStderr(self, job):
    return CheckRemoteCommands(job.host, ["cat '%s'" % os.path.join(
        self.results_dir, job.id_, "err")], verbose = self.verbose)

  def GetStdout(self, job):
    return CheckRemoteCommands(job.host, ["cat '%s'" % os.path.join(
        self.results_dir, job.id_, "log")], verbose = self.verbose)

  def LaunchJob(self, job, host):
    """Start a job on a remote host, returning the allocated job ID.
    Requires that the command 'gjob' be on the remote path of the worker node. The set of job commands is launched as a
    bash script.
    """
    # Make experiment directory, recording experiment ID.
    job_id = CheckRemoteCommands(host,
        ["cd '%s'" % self.results_dir, "gjob local mkdir"], self.verbose)
    job_id = job_id.strip()
    exp_dir = os.path.join(self.results_dir, job_id)
    # Copy experimental files to remote experiment directory.
    CopyLocalFilesToRemoteHost(host, exp_dir, *job.spec.files,
        verbose = self.verbose)
    WriteRemoteFile(host, os.path.join(exp_dir, ".cmds"),
        "\n".join(job.spec.cmds) + "\n", self.verbose)
    cmd = "gjob local start '%s' bash .cmds 1>%s/log 2>%s/err &" % (exp_dir,
        exp_dir, exp_dir)
    CheckRemoteCommands(host, [cmd], self.verbose)
    return job_id

def CopyLocalFilesToRemoteHost(host, remote_path, *files, **check_opts):
  """Copy a set of files to a remote node.
  host -- name of remote node
  remote_path -- path of remote directory to which local files are copied
  files -- set of local files to copy
  check_opts -- optional arguments for CheckLocalCommand()
  Requires that the command 'scp' be on the local path.
  """
  if len(files) < 1:
    return
  if check_opts.get('verbose', False):
    print "Writing %s to '%s' on '%s'" % (files, remote_path, host)
  for local_path in files:
    assert(os.path.exists(local_path)), "Local file not found: %s" % local_path
  CheckLocalCommand(["scp", "-q"] + list(files) + ["%s:%s" % (host,
      remote_path)], **check_opts)

def WriteRemoteFile(host, remote_path, contents, verbose = False):
  """Write data stored in memory to a file on a remote node.
  host -- name of remote node
  remote_path -- path of output file on remote node
  contents -- data to write
  verbose -- flag controlling whether extra logging information is printed
  Requires that the command 'ssh' be on the local path.
  """
  if verbose:
    print "Writing to '%s' on '%s'\n%s" % (remote_path, host, contents)
  p = subprocess.Popen(("ssh", "-q", host, "cat > %s" % remote_path),
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = p.communicate(contents)
  if not (stderr == None or stderr == ""):
    raise Exception("Remote write failed to '%s' on '%s'\n%s" % (remote_path,
        host, stderr))
  return stdout

def CheckRemoteCommands(host, cmds, verbose = False):
  """Open a pipe via SSH, execute given commands, close the pipe, and return
  stdout.
  host -- name of remote node
  cmds -- (string list) commands to execute on remote node
  verbose -- flag controlling whether extra logging information is printed
  Requires that the command 'ssh' be on the local path.
  """
  if verbose:
    print "Running remote commands on '%s'\n%s" % (host,
        "\t" + "\n\t".join(cmds))
  p = subprocess.Popen(("ssh", "-q", host), stdin=subprocess.PIPE,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = p.communicate("\n".join(cmds))
  if not (stderr == None or stderr == ""):
    raise Exception("SSH command failed on host %s: (stderr below)\n%s\n(cmds "
        "below)\n%s" % (host, stderr, "\n".join(cmds)))
  return stdout

def CheckLocalCommand(cmd, verbose = False):
  """Run a single command on local machine, returning stdout. Throw an exception
  if the command failed.
  cmd -- (string list) command to run
  verbose -- flag controlling whether extra logging information is printed
  """
  if verbose:
    print "Running local command: '%s'" % " ".join(cmd)
  retcode = subprocess.call(cmd)
  if retcode != 0:
    raise Exception("Local command failed: '%s'" % " ".join(cmd))

class Manager(object):
  """Allocates N jobs to M hosts, with N > M."""

  def __init__(self, sleep_time_in_secs):
    self.sleep_time_in_secs = sleep_time_in_secs
    self.jobs = None
    self.hosts = None
    self.network = None

  def Setup(self, job_specs, network):
    """Initialize the manager to process a set of jobs.
    jobs_specs -- (JobSpec list) Command, arguments, and files required to
                  launch each of a set of jobs.
    network -- (Network) The network on which to launch the jobs.
    """
    jobs = [ Job(spec) for spec in job_specs ]
    self.jobs = Queue([STATE_FREE, STATE_BUSY, STATE_DONE], jobs)
    self.network = network
    self.hosts = Queue([STATE_FREE, STATE_BUSY], network.GetHosts())

  def LaunchJob(self):
    """Launch the next job in the queue on a free host."""
    job = self.jobs.ChooseNext(STATE_FREE)
    host = self.hosts.ChooseRandom(STATE_FREE)
    job.id_ = self.network.LaunchJob(job, host)
    job.host = host
    self.jobs.SetState(job, STATE_BUSY)
    self.hosts.SetState(host, STATE_BUSY)
    self.HandleJobLaunch(job)

  def HandleJobLaunch(self, job):
    """Event handler -- called when job begins."""
    pass

  def HandleJobDone(self, job):
    """Event handler -- called when job completes."""
    pass

  def HandleSleep(self):
    """Event handler -- called when manager waits on finished jobs."""
    pass

  def UpdateJobStatus(self):
    """Poll hosts for job status, returning True if any job has finished."""
    busy_jobs = self.jobs.InState(STATE_BUSY)
    results = self.network.GetJobStatus(busy_jobs)
    event_fired = False
    for job, state in zip(busy_jobs, results):
      if state == STATE_DONE:
        self.jobs.SetState(job, STATE_DONE)
        self.hosts.SetState(job.host, STATE_FREE)
        self.HandleJobDone(job)
        event_fired = True
      elif state == STATE_ERROR:
        raise Exception("Job (%s) had error (below):\n%s" % (job.id_,
            self.network.GetStderr(job)))
      elif state != STATE_BUSY:
        raise Exception("Job (%s) in unexpected state: %s" % (job.id_, state))
    return event_fired

  def ProcessJobs(self):
    """Run all jobs in the queue, blocking until they finish."""
    while not self.jobs.AllInState(STATE_DONE):
      if self.jobs.AnyInState(STATE_FREE) and self.hosts.AnyInState(STATE_FREE):
        self.LaunchJob()
      else:
        job_finished = self.UpdateJobStatus()
        if not job_finished:
          self.HandleSleep()
          time.sleep(self.sleep_time_in_secs)

class LoggingManager(Manager):
  """Manager that logs job events to disk."""

  def __init__(self, sleep_time_in_secs, log):
    Manager.__init__(self, sleep_time_in_secs)
    self.log = log
    self.last_was_sleep = False

  def HandleJobLaunch(self, job):
    if self.last_was_sleep:
      self.log.write("\n")
      self.last_was_sleep = False
    idx = self.jobs.objects.index(job)
    self.log.write("LAUNCH %s w/name %s as job #%d\n" % (job.id_, job.spec.name,
        idx))
    self.log.flush()

  def HandleJobDone(self, job):
    if self.last_was_sleep:
      self.log.write("\n")
      self.last_was_sleep = False
    idx = self.jobs.objects.index(job)
    self.log.write("DONE %s w/name %s as job #%d\n" % (job.id_, job.spec.name,
        idx))
    self.log.flush()

  def HandleSleep(self):
    if self.last_was_sleep:
      self.log.write(".")
    else:
      self.log.write("WAIT .")
      self.last_was_sleep = True
    self.log.flush()

class JobSpec(object):
  """Describes the commands necessary to launch a job."""

  def __init__(self, cmds, files = [], name = None):
    assert(len(cmds) > 0), "Must specify at least one command"
    self.cmds = cmds
    self.files = files
    self.name = name or "no-name"

  def __str__(self):
    return "JobSpec(name: %s, cmds: %s, files: %s)" % (self.name, self.cmds,
        self.files)

  __repr__ = __str__

class Job(object):
  """The record of a job, including its spec and ID."""

  def __init__(self, job_spec):
    self.spec = job_spec
    self.id_ = None
    self.host = None

  def __str__(self):
    return "Job(id: %s, host: %s, spec: %s)" % (self.id_, self.host, self.spec)

class Builder(object):
  """Provides a builder syntax for constructing job lists."""

  def __init__(self):
    self.repeat = 1
    self.job_specs = []

  def AddJob(self, **args):
    for i in range(args.get('repeat', self.repeat)):
      self.job_specs.append( JobSpec(args.get('commands', []), args.get('files',
          []), args.get('name', None)) )

  def SetRepeat(self, repeat):
    self.repeat = repeat

  def MakeJobSpecs(self):
    """Generate the list of job specs described by previous calls to this
    object."""
    return self.job_specs
