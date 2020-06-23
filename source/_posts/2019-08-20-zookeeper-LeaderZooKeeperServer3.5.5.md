---
layout:     post
title:      "zookeeper-LeaderZooKeeperServer3.5.5"
date:       2019-08-20
author:     "Gary"
header-img: "img/post-bg-unix-linux.jpg"
tags:
---
[toc]
# LeaderZooKeeperServer

```java
    CommitProcessor commitProcessor;
    PrepRequestProcessor prepRequestProcessor;
    protected void setupRequestProcessors() {
        RequestProcessor finalProcessor = new FinalRequestProcessor(this);
        RequestProcessor toBeAppliedProcessor = new Leader.ToBeAppliedRequestProcessor(finalProcessor, getLeader());
        commitProcessor = new CommitProcessor(toBeAppliedProcessor,
                Long.toString(getServerId()), false,
                getZooKeeperServerListener());
        commitProcessor.start();
        ProposalRequestProcessor proposalProcessor = new ProposalRequestProcessor(this,
                commitProcessor);
        proposalProcessor.initialize();
        prepRequestProcessor = new PrepRequestProcessor(this, proposalProcessor);
        prepRequestProcessor.start();
        firstProcessor = new LeaderRequestProcessor(this, prepRequestProcessor);

        setupContainerManager();
    }
```
可以看到有3个处理链:
1. `firstProcessor`处理链`LeaderRequestProcessor->PrepRequestProcessor->ProposalRequestProcessor->CommitProcessor->Leader.ToBeAppliedRequestProcessor ->FinalRequestProcessor`
2. `CommitProcessor->Leader.ToBeAppliedRequestProcessor ->FinalRequestProcessor`
3. `PrepRequestProcessor->ProposalRequestProcessor->CommitProcessor->Leader.ToBeAppliedRequestProcessor ->FinalRequestProcessor`

当外部请求来时，使用的是`firstProcessor`。

## 1 LeaderRequestProcessor

```java
    @Override
    public void processRequest(Request request)
            throws RequestProcessorException {
        // Check if this is a local session and we are trying to create
        // an ephemeral node, in which case we upgrade the session
        Request upgradeRequest = null;
        try {
            upgradeRequest = lzks.checkUpgradeSession(request);
        } catch (KeeperException ke) {
            if (request.getHdr() != null) {
                LOG.debug("Updating header");
                request.getHdr().setType(OpCode.error);
                request.setTxn(new ErrorTxn(ke.code().intValue()));
            }
            request.setException(ke);
            LOG.info("Error creating upgrade request " + ke.getMessage());
        } catch (IOException ie) {
            LOG.error("Unexpected error in upgrade", ie);
        }
        if (upgradeRequest != null) {
            nextProcessor.processRequest(upgradeRequest);
        }

        nextProcessor.processRequest(request);
    }
```

看英文注释，`LeaderRequestProcessor`只是更新了下session，然后交给下个来处理。

`PrepRequestProcessor`之前分析过，这里不写了。

## 2 ProposalRequestProcessor

```java
    public ProposalRequestProcessor(LeaderZooKeeperServer zks,
            RequestProcessor nextProcessor) {
        this.zks = zks;
        this.nextProcessor = nextProcessor;
        AckRequestProcessor ackProcessor = new AckRequestProcessor(zks.getLeader());
        syncProcessor = new SyncRequestProcessor(zks, ackProcessor);
    }

    public void processRequest(Request request) throws RequestProcessorException {
        //如果是LearnerSyncRequest，调用leader的processSync方法，应该跟follow同步有关
        if (request instanceof LearnerSyncRequest){
            zks.getLeader().processSync((LearnerSyncRequest)request);
        } else {
            //交给下一个处理器处理，如果是事务操作的话（也就是写操作，个人理解），调用leader.propose和syncProcessor
            nextProcessor.processRequest(request);
            if (request.getHdr() != null) {
                // We need to sync and get consensus on any transactions
                try {
                    zks.getLeader().propose(request);
                } catch (XidRolloverException e) {
                    throw new RequestProcessorException(e.getMessage(), e);
                }
                //如果是事物消息的话，调用syncProcessor
                syncProcessor.processRequest(request);
            }
        }
    }
```

`ProposalRequestProcessor`的功能总结：
1. 如果是sync请求，调用leader.processSync。
2. 非sync请求，调用

所以请求的调用链可以调整为：
```
PrepRequestProcessor->ProposalRequestProcessor->
                                          CommitProcessor->ToBeAppliedRequestProcessor->FinalRequestProcessor
                                          SyncRequestProcessor->AckRequestProcessor
```



## 3. CommitProcessor

```java

    /**
     * Requests that we are holding until the commit comes in.
     */
    protected final LinkedBlockingQueue<Request> queuedRequests =
        new LinkedBlockingQueue<Request>();
    /**
     * Requests that have been committed.
     */
    protected final LinkedBlockingQueue<Request> committedRequests =
        new LinkedBlockingQueue<Request>();
    /** Request for which we are currently awaiting a commit */
    protected final AtomicReference<Request> nextPending =
        new AtomicReference<Request>();
    /** Request currently being committed (ie, sent off to next processor) */
    private final AtomicReference<Request> currentlyCommitting =
        new AtomicReference<Request>();
    @Override
    public void processRequest(Request request) {
        if (stopped) {
            return;
        }
        if (LOG.isDebugEnabled()) {
            LOG.debug("Processing request:: " + request);
        }
        //把request放入queuedRequests，等待处理
        queuedRequests.add(request);
        if (!isWaitingForCommit()) {
            wakeup();
        }
    }
    private boolean isWaitingForCommit() {
        return nextPending.get() != null;
    }

    private boolean isProcessingCommit() {
        return currentlyCommitting.get() != null;
    }
    @Override
    public void run() {
        Request request;
        try {
            while (!stopped) {
                //如果queuedRequests和committedRequests都是空 && 有正在处理的请求，一直等待
                synchronized(this) {
                    while (
                        !stopped &&
                        ((queuedRequests.isEmpty() || isWaitingForCommit() || isProcessingCommit()) &&
                         (committedRequests.isEmpty() || isProcessingRequest()))) {
                        wait();
                    }
                }

                /*
                 * Processing queuedRequests: Process the next requests until we
                 * find one for which we need to wait for a commit. We cannot
                 * process a read request while we are processing write request.
                 */
                 //nextPending和currentlyCommitting没数据，且queuedRequests有数据。
                while (!stopped && !isWaitingForCommit() &&
                       !isProcessingCommit() &&
                       (request = queuedRequests.poll()) != null) {
                    //如果是写请求，写入nextPending，WaitingForCommit
                    if (needCommit(request)) {
                        nextPending.set(request);
                    } else {
                        //交给下面处理器处理
                        sendToNextProcessor(request);
                    }
                }

                /*
                 * Processing committedRequests: check and see if the commit
                 * came in for the pending request. We can only commit a
                 * request when there is no other request being processed.
                 */
                processCommitted();
            }
        } catch (Throwable e) {
            handleException(this.getName(), e);
        }
        LOG.info("CommitProcessor exited loop!");
    }
    
    //commit方法，由leader调用，将commit请求放入队列committedRequests
    public void commit(Request request) {
        if (stopped || request == null) {
            return;
        }
        if (LOG.isDebugEnabled()) {
            LOG.debug("Committing request:: " + request);
        }
        committedRequests.add(request);
        if (!isProcessingCommit()) {
            wakeup();
        }
    }
    
    protected void processCommitted() {
        Request request;

        if (!stopped && !isProcessingRequest() &&
                (committedRequests.peek() != null)) {

            /*
             * ZOOKEEPER-1863: continue only if there is no new request
             * waiting in queuedRequests or it is waiting for a
             * commit. 
             */
            if ( !isWaitingForCommit() && !queuedRequests.isEmpty()) {
                return;
            }
            //从committedRequests中取出请求
            request = committedRequests.poll();

            /*
             * We match with nextPending so that we can move to the
             * next request when it is committed. We also want to
             * use nextPending because it has the cnxn member set
             * properly.
             */
             //commitRequest和pendingRequest进行比对。但是最后都是交给下面的处理，好像并没有什么区别。
            Request pending = nextPending.get();
            if (pending != null &&
                pending.sessionId == request.sessionId &&
                pending.cxid == request.cxid) {
                // we want to send our version of the request.
                // the pointer to the connection in the request
                pending.setHdr(request.getHdr());
                pending.setTxn(request.getTxn());
                pending.zxid = request.zxid;
                // Set currentlyCommitting so we will block until this
                // completes. Cleared by CommitWorkRequest after
                // nextProcessor returns.
                currentlyCommitting.set(pending);
                nextPending.set(null);
                sendToNextProcessor(pending);
            } else {
                // this request came from someone else so just
                // send the commit packet
                currentlyCommitting.set(request);
                sendToNextProcessor(request);
            }
        }      
    }
```

## 4 ToBeAppliedRequestProcessor

`ToBeAppliedRequestProcessor`是`Leader`的内部类。

```java
    static class ToBeAppliedRequestProcessor implements RequestProcessor {
        private final RequestProcessor next;

        private final Leader leader;

        /**
         * This request processor simply maintains the toBeApplied list. For
         * this to work next must be a FinalRequestProcessor and
         * FinalRequestProcessor.processRequest MUST process the request
         * synchronously!
         *
         * @param next
         *                a reference to the FinalRequestProcessor
         */
        ToBeAppliedRequestProcessor(RequestProcessor next, Leader leader) {
            if (!(next instanceof FinalRequestProcessor)) {
                throw new RuntimeException(ToBeAppliedRequestProcessor.class
                        .getName()
                        + " must be connected to "
                        + FinalRequestProcessor.class.getName()
                        + " not "
                        + next.getClass().getName());
            }
            this.leader = leader;
            this.next = next;
        }

        /*
         * (non-Javadoc)
         *
         * @see org.apache.zookeeper.server.RequestProcessor#processRequest(org.apache.zookeeper.server.Request)
         */
        public void processRequest(Request request) throws RequestProcessorException {
            next.processRequest(request);

            // The only requests that should be on toBeApplied are write
            // requests, for which we will have a hdr. We can't simply use
            // request.zxid here because that is set on read requests to equal
            // the zxid of the last write op.
            if (request.getHdr() != null) {
                long zxid = request.getHdr().getZxid();
                Iterator<Proposal> iter = leader.toBeApplied.iterator();
                if (iter.hasNext()) {
                    Proposal p = iter.next();
                    if (p.request != null && p.request.zxid == zxid) {
                        iter.remove();
                        return;
                    }
                }
                LOG.error("Committed request not found on toBeApplied: "
                          + request);
            }
        }

        /*
         * (non-Javadoc)
         *
         * @see org.apache.zookeeper.server.RequestProcessor#shutdown()
         */
        public void shutdown() {
            LOG.info("Shutting down");
            next.shutdown();
        }
    }

```

`ToBeAppliedRequestProcessor`的作用就是用下个处理器处理，然后把`request`从`toBeApplied`队列里删掉。

## 5.AckRequestProcessor

```java
class AckRequestProcessor implements RequestProcessor {
    private static final Logger LOG = LoggerFactory.getLogger(AckRequestProcessor.class);
    Leader leader;

    AckRequestProcessor(Leader leader) {
        this.leader = leader;
    }

    /**
     * Forward the request as an ACK to the leader
     */
    public void processRequest(Request request) {
        QuorumPeer self = leader.self;
        if(self != null)
            leader.processAck(self.getId(), request.zxid, null);
        else
            LOG.error("Null QuorumPeer");
    }

    public void shutdown() {
        // XXX No need to do anything
    }
}
```

`AckRequestProcessor`的作用就是调`leader.processAck`来处理请求。

## 6.leader

`leader`代码有点多，主要看下跟投票相关的核心代码吧。
先从`LearnerHandler`开始，`leader`启动时会对每个`Quorum`节点建立socket连接，`LearnerHandler`就是用来处理这个连接的。

```java
switch (qp.getType()) {
                case Leader.ACK:
                    if (this.learnerType == LearnerType.OBSERVER) {
                        if (LOG.isDebugEnabled()) {
                            LOG.debug("Received ACK from Observer  " + this.sid);
                        }
                    }
                    syncLimitCheck.updateAck(qp.getZxid());
                    leader.processAck(this.sid, qp.getZxid(), sock.getLocalSocketAddress());
                    break;
                case Leader.PING:
                    // Process the touches
                    ByteArrayInputStream bis = new ByteArrayInputStream(qp
                            .getData());
                    DataInputStream dis = new DataInputStream(bis);
                    while (dis.available() > 0) {
                        long sess = dis.readLong();
                        int to = dis.readInt();
                        leader.zk.touch(sess, to);
                    }
                    break;
                case Leader.REVALIDATE:
                    bis = new ByteArrayInputStream(qp.getData());
                    dis = new DataInputStream(bis);
                    long id = dis.readLong();
                    int to = dis.readInt();
                    ByteArrayOutputStream bos = new ByteArrayOutputStream();
                    DataOutputStream dos = new DataOutputStream(bos);
                    dos.writeLong(id);
                    boolean valid = leader.zk.checkIfValidGlobalSession(id, to);
                    if (valid) {
                        try {
                            //set the session owner
                            // as the follower that
                            // owns the session
                            leader.zk.setOwner(id, this);
                        } catch (SessionExpiredException e) {
                            LOG.error("Somehow session " + Long.toHexString(id) +
                                    " expired right after being renewed! (impossible)", e);
                        }
                    }
                    if (LOG.isTraceEnabled()) {
                        ZooTrace.logTraceMessage(LOG,
                                                 ZooTrace.SESSION_TRACE_MASK,
                                                 "Session 0x" + Long.toHexString(id)
                                                 + " is valid: "+ valid);
                    }
                    dos.writeBoolean(valid);
                    qp.setData(bos.toByteArray());
                    queuedPackets.add(qp);
                    break;
                case Leader.REQUEST:
                    bb = ByteBuffer.wrap(qp.getData());
                    sessionId = bb.getLong();
                    cxid = bb.getInt();
                    type = bb.getInt();
                    bb = bb.slice();
                    Request si;
                    if(type == OpCode.sync){
                        si = new LearnerSyncRequest(this, sessionId, cxid, type, bb, qp.getAuthinfo());
                    } else {
                        si = new Request(null, sessionId, cxid, type, bb, qp.getAuthinfo());
                    }
                    si.setOwner(this);
                    leader.zk.submitLearnerRequest(si);
                    break;
                default:
                    LOG.warn("unexpected quorum packet, type: {}", packetToString(qp));
                    break;
                }
```

可以看到分别对请求类型做了不同的处理：

* Leader.ACK -> `leader.processAck`
* Leader.PING -> `leader.zk.touch`
* Leader.REVALIDATE -> `leader.zk.checkIfValidGlobalSession`和`leader.zk.setOwner`
* Leader.REQUEST -> `leader.zk.submitLearnerRequest`

`Leader.ACK`应该是follower对请求的反馈。
`Leader.REQUEST`应该是follower转发的请求，而`leader.zk.submitLearnerRequest`内部实现其实是调用链。上面分析得知`ProposalRequestProcessor`在处理Proposal时会调用`leader.propose`方法，所以我们重点关注两个方法:

 * `leader.propose`处理proposal
 * `leader.processAck`处理proposal ack

```java
    public Proposal propose(Request request) throws XidRolloverException {
        /**
         * Address the rollover issue. All lower 32bits set indicate a new leader
         * election. Force a re-election instead. See ZOOKEEPER-1277
         */
         //ZXID的后32位如果是0xffffffff，关闭
        if ((request.zxid & 0xffffffffL) == 0xffffffffL) {
            String msg =
                    "zxid lower 32 bits have rolled over, forcing re-election, and therefore new epoch start";
            shutdown(msg);
            throw new XidRolloverException(msg);
        }

        byte[] data = SerializeUtils.serializeRequest(request);
        proposalStats.setLastBufferSize(data.length);
        //构建QuorumPacket和Proposal
        QuorumPacket pp = new QuorumPacket(Leader.PROPOSAL, request.zxid, data, null);

        Proposal p = new Proposal();
        p.packet = pp;
        p.request = request;                
        
        synchronized(this) {
           p.addQuorumVerifier(self.getQuorumVerifier());
                   
           if (request.getHdr().getType() == OpCode.reconfig){
               self.setLastSeenQuorumVerifier(request.qv, true);                       
           }
           
           if (self.getQuorumVerifier().getVersion()<self.getLastSeenQuorumVerifier().getVersion()) {
               p.addQuorumVerifier(self.getLastSeenQuorumVerifier());
           }
                   
            if (LOG.isDebugEnabled()) {
                LOG.debug("Proposing:: " + request);
            }

            lastProposed = p.packet.getZxid();
            //写入outstandingProposals队列然后发送
            outstandingProposals.put(lastProposed, p);
            sendPacket(pp);
        }
        return p;
    }
```

```java
    void sendPacket(QuorumPacket qp) {
        //往所有的follower发送数据
        synchronized (forwardingFollowers) {
            for (LearnerHandler f : forwardingFollowers) {
                //LearnerHandler其实也是将数据放入队列中。
                f.queuePacket(qp);
            }
        }
    }
```

`propose`方法很简单，构建`QuorumPacket`，放入HashMap `outstandingProposals`，然后广播。后面看ack。

```java
    /**
     * Keep a count of acks that are received by the leader for a particular
     * proposal
     *
     * @param zxid, the zxid of the proposal sent out
     * @param sid, the id of the server that sent the ack
     * @param followerAddr
     */
    synchronized public void processAck(long sid, long zxid, SocketAddress followerAddr) {        
        if (!allowedToCommit) return; // last op committed was a leader change - from now on 
                                     // the new leader should commit        
        if (LOG.isTraceEnabled()) {
            LOG.trace("Ack zxid: 0x{}", Long.toHexString(zxid));
            for (Proposal p : outstandingProposals.values()) {
                long packetZxid = p.packet.getZxid();
                LOG.trace("outstanding proposal: 0x{}",
                        Long.toHexString(packetZxid));
            }
            LOG.trace("outstanding proposals all");
        }
        
        if ((zxid & 0xffffffffL) == 0) {
            /*
             * We no longer process NEWLEADER ack with this method. However,
             * the learner sends an ack back to the leader after it gets
             * UPTODATE, so we just ignore the message.
             */
            return;
        }
            
        //如果没有在外的proposal，直接返回
        if (outstandingProposals.size() == 0) {
            if (LOG.isDebugEnabled()) {
                LOG.debug("outstanding is 0");
            }
            return;
        }
        //如果请求的zxid小于最后一次提交的zxid，表示这个proposal已经提交过了，忽略这个请求。
        if (lastCommitted >= zxid) {
            if (LOG.isDebugEnabled()) {
                LOG.debug("proposal has already been committed, pzxid: 0x{} zxid: 0x{}",
                        Long.toHexString(lastCommitted), Long.toHexString(zxid));
            }
            // The proposal has already been committed
            return;
        }
        //从outstandingProposals获取具体的proposal对象
        Proposal p = outstandingProposals.get(zxid);
        if (p == null) {
            LOG.warn("Trying to commit future proposal: zxid 0x{} from {}",
                    Long.toHexString(zxid), followerAddr);
            return;
        }
        //记录ack
        p.addAck(sid);        
        /*if (LOG.isDebugEnabled()) {
            LOG.debug("Count for zxid: 0x{} is {}",
                    Long.toHexString(zxid), p.ackSet.size());
        }*/
        //尝试提交
        boolean hasCommitted = tryToCommit(p, zxid, followerAddr);

        // If p is a reconfiguration, multiple other operations may be ready to be committed,
        // since operations wait for different sets of acks.
       // Currently we only permit one outstanding reconfiguration at a time
       // such that the reconfiguration and subsequent outstanding ops proposed while the reconfig is
       // pending all wait for a quorum of old and new config, so its not possible to get enough acks
       // for an operation without getting enough acks for preceding ops. But in the future if multiple
       // concurrent reconfigs are allowed, this can happen and then we need to check whether some pending
        // ops may already have enough acks and can be committed, which is what this code does.

        if (hasCommitted && p.request!=null && p.request.getHdr().getType() == OpCode.reconfig){
               long curZxid = zxid;
           while (allowedToCommit && hasCommitted && p!=null){
               curZxid++;
               p = outstandingProposals.get(curZxid);
               if (p !=null) hasCommitted = tryToCommit(p, curZxid, null);             
           }
        }
    }
```

`processAck` 先是对zxid进行校验，然后记录，最后调用`tryToCommit`尝试提交。



```java
    synchronized public boolean tryToCommit(Proposal p, long zxid, SocketAddress followerAddr) {       
       // make sure that ops are committed in order. With reconfigurations it is now possible
       // that different operations wait for different sets of acks, and we still want to enforce
       // that they are committed in order. Currently we only permit one outstanding reconfiguration
       // such that the reconfiguration and subsequent outstanding ops proposed while the reconfig is
       // pending all wait for a quorum of old and new config, so it's not possible to get enough acks
       // for an operation without getting enough acks for preceding ops. But in the future if multiple
       // concurrent reconfigs are allowed, this can happen.
       //保证顺序
       if (outstandingProposals.containsKey(zxid - 1)) return false;
       
       // in order to be committed, a proposal must be accepted by a quorum.
       //
       // getting a quorum from all necessary configurations.
       //过半提交
        if (!p.hasAllQuorums()) {
           return false;                 
        }
        
        // commit proposals in order
        if (zxid != lastCommitted+1) {    
           LOG.warn("Commiting zxid 0x" + Long.toHexString(zxid)
                    + " from " + followerAddr + " not first!");
            LOG.warn("First is "
                    + (lastCommitted+1));
        }     
        //从outstandingProposals中移除
        outstandingProposals.remove(zxid);
        //toBeApplied队列加入
        if (p.request != null) {
             toBeApplied.add(p);
        }

        if (p.request == null) {
            LOG.warn("Going to commmit null: " + p);
        } else if (p.request.getHdr().getType() == OpCode.reconfig) {                                   
            LOG.debug("Committing a reconfiguration! " + outstandingProposals.size()); 
                 
            //if this server is voter in new config with the same quorum address, 
            //then it will remain the leader
            //otherwise an up-to-date follower will be designated as leader. This saves
            //leader election time, unless the designated leader fails                             
            Long designatedLeader = getDesignatedLeader(p, zxid);
            //LOG.warn("designated leader is: " + designatedLeader);

            QuorumVerifier newQV = p.qvAcksetPairs.get(p.qvAcksetPairs.size()-1).getQuorumVerifier();
       
            self.processReconfig(newQV, designatedLeader, zk.getZxid(), true);

            if (designatedLeader != self.getId()) {
                allowedToCommit = false;
            }
                   
            // we're sending the designated leader, and if the leader is changing the followers are 
            // responsible for closing the connection - this way we are sure that at least a majority of them 
            // receive the commit message.
            commitAndActivate(zxid, designatedLeader);
            informAndActivate(p, designatedLeader);
            //turnOffFollowers();
        } else {
            //向所有follower发送commit请求。
            commit(zxid);
            //向所有Observer广播
            inform(p);
        }
        //commitProcessor处理commit请求。
        zk.commitProcessor.commit(p.request);
        if(pendingSyncs.containsKey(zxid)){
            for(LearnerSyncRequest r: pendingSyncs.remove(zxid)) {
                sendSync(r);
            }               
        } 
        
        return  true;   
    }
    public void commit(long zxid) {
        synchronized(this){
            lastCommitted = zxid;
        }
        QuorumPacket qp = new QuorumPacket(Leader.COMMIT, zxid, null, null);
        sendPacket(qp);
    }
    public void inform(Proposal proposal) {
        QuorumPacket qp = new QuorumPacket(Leader.INFORM, proposal.request.zxid,
                                            proposal.packet.getData(), null);
        sendObserverPacket(qp);
    }
```

## 7. leader propose流程总结

* 1. 请求处理入口有2个，一个是`org.apache.zookeeper.server.ZooKeeperServer#submitRequest`，另外一个是`LearnerHandler`的follower转发请求，都是调用处理链来处理。
* 2. 处理链总结为:PrepRequestProcessor->ProposalRequestProcessor->
                                          CommitProcessor->ToBeAppliedRequestProcessor->FinalRequestProcessor
                                          SyncRequestProcessor->AckRequestProcessor
* 3. ProposalRequestProcessor调用leader的propose方法广播proposal，同时调用2个处理链。
* 4. 处理链`SyncRequestProcessor->AckRequestProcessor`可以理解为本地的proposal处理，SyncRequestProcessor将proposal存入磁盘，AckRequestProcessor会调用leader.processAck来返回ack。
* 5. 处理链`CommitProcessor->ToBeAppliedRequestProcessor->FinalRequestProcessor`，CommitProcessor会阻塞住一个请求，直到被调用commit方法，然后交由FinalRequestProcessor修改内存数据。
* 6. leader和follower处理完后，会调用`org.apache.zookeeper.server.quorum.Leader#processAck`，leader判断获得半数以上的反馈后，会调用CommitProcessor的commit方法，跳转到5。


## 8. FollowerZooKeeperServer

理解了LeaderZooKeeperServer之后再来看FollowerZooKeeperServer，发现其实很好理解。

```java
    @Override
    protected void setupRequestProcessors() {
        RequestProcessor finalProcessor = new FinalRequestProcessor(this);
        commitProcessor = new CommitProcessor(finalProcessor,
                Long.toString(getServerId()), true, getZooKeeperServerListener());
        commitProcessor.start();
        firstProcessor = new FollowerRequestProcessor(this, commitProcessor);
        ((FollowerRequestProcessor) firstProcessor).start();
        syncProcessor = new SyncRequestProcessor(this,
                new SendAckRequestProcessor((Learner)getFollower()));
        syncProcessor.start();
    }

    LinkedBlockingQueue<Request> pendingTxns = new LinkedBlockingQueue<Request>();

    //处理proposal
    public void logRequest(TxnHeader hdr, Record txn) {
        Request request = new Request(hdr.getClientId(), hdr.getCxid(), hdr.getType(), hdr, txn, hdr.getZxid());
        if ((request.zxid & 0xffffffffL) != 0) {
            pendingTxns.add(request);
        }
        syncProcessor.processRequest(request);
    }

    /**
     * When a COMMIT message is received, eventually this method is called,
     * which matches up the zxid from the COMMIT with (hopefully) the head of
     * the pendingTxns queue and hands it to the commitProcessor to commit.
     * @param zxid - must correspond to the head of pendingTxns if it exists
     */
     //处理commit请求
    public void commit(long zxid) {
        if (pendingTxns.size() == 0) {
            LOG.warn("Committing " + Long.toHexString(zxid)
                    + " without seeing txn");
            return;
        }
        long firstElementZxid = pendingTxns.element().zxid;
        if (firstElementZxid != zxid) {
            LOG.error("Committing zxid 0x" + Long.toHexString(zxid)
                    + " but next pending txn 0x"
                    + Long.toHexString(firstElementZxid));
            System.exit(12);
        }
        Request request = pendingTxns.remove();
        commitProcessor.commit(request);
    }
```


两个处理链：

* 1.FollowerRequestProcessor->CommitProcessor->FinalRequestProcessor
* 2.SyncRequestProcessor->SendAckRequestProcessor

proposal请求时，调用处理链`SyncRequestProcessor->SendAckRequestProcessor`，写入磁盘并返回ack。
commit时，调用处理链`CommitProcessor->FinalRequestProcessor`，写入内存。

