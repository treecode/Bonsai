#include <iostream>
#include <omp.h>
#include "Particle.h"
#include "boundary.h"

static inline float lWkernel(const float q)
{
  const float sigma = 8.0f/M_PI;

  const float qm = 1.0f - q;
  if      (q < 0.5f) return sigma * (1.0f + (-6.0f)*q*q*qm);
  else if (q < 1.0f) return sigma * 2.0f*qm*qm*qm;

  return 0.0f;
}

static inline void lInteract(Particle &ip, Particle &jp, const float r2)
{
  const float hinv = ip.get_hinv();
  const float r    = std::sqrt(r2);
  const float q    = r * hinv;
  const float hinv3 = hinv*hinv*hinv;

  ip.density += jp.mass * lWkernel(q) * hinv3;
}


struct Node
{
  static const int NLEAF = 32;
  static std::vector<Particle> ptcl;
  static std::vector<Node> Node_heap;
  static std::vector<std::pair<Node*, Node*> > pair_list;
  static void allocate(int nptcl, int nNode){
    ptcl.reserve(nptcl);
    Node_heap.reserve(nNode);
  }
  static void clear(){
    ptcl.clear();
    Node_heap.clear();
  }
  typedef boundary<float> Boundary;

  int np;     // number of Particle;
  int depth;
  float size;
  int pfirst; // first Particle
  int cfirst; // first child
  Boundary bound_inner;
  Boundary bound_outer;

  Node()           : np(0), depth(     0), pfirst(-1), cfirst(-1) {}
  Node(int _depth, float _size) : np(0), depth(_depth), size(_size), pfirst(-1), cfirst(-1) {}

  bool is_leaf() const
  {
    return np < NLEAF;
  }
  void push_particle(const int paddr, const int rshift)
  {
    assert(rshift >= 0);
    if(!is_leaf())
    { // assign recursively
      int ic = ptcl[paddr].octkey(rshift);
      Node &child = Node_heap[cfirst + ic];
      child.push_particle(paddr, rshift-3);
    }
    else
    {
      if(-1 == pfirst)
      {
        assert(0 == np);
        pfirst = paddr;
      }
    }
    np++;
    if(np == NLEAF)
    { // shi's just become a mother
      assert(pfirst >= 0);
      cfirst = Node_heap.size();
#if 0
      for(int ic=0; ic<8; ic++){
        Node_heap.push_back(Node(1+depth));
      }
#else
      size_t new_size = Node_heap.size() + 8;
      assert(Node_heap.capacity() >= new_size);
      Node_heap.resize(new_size, Node(1+depth,size/2.0));
#endif
      for(int addr = pfirst; addr < pfirst+np; addr++)
      {
        int ic = ptcl[addr].octkey(rshift);
        Node &child = Node_heap[cfirst + ic];
        child.push_particle(addr, rshift-3);
      }
    }
  }
  void dump_tree(
      int level,
      std::ostream &ofs = std::cout) const{
    if(is_leaf()){
      for(int ip=0; ip<np; ip++){
        const Particle &p = ptcl[ip+pfirst];
        for(int i=0; i<level; i++) ofs << " ";
        ofs << p.pos << std::endl;
      }
      ofs << std::endl;
    }else{
      for(int i=0; i<level; i++) ofs << ">";
      ofs << std::endl;
      for(int ic=0; ic<8; ic++){
        const Node &child = Node_heap[cfirst + ic];
        child.dump_tree(level+1, ofs);
      }
    }
  }
  void make_boundary(){
    if(is_leaf()){
      for(int ip=0; ip<np; ip++){
        const Particle &p = ptcl[ip+pfirst];
        bound_inner.merge(Boundary(p.pos));
        bound_outer.merge(Boundary(p.pos, p.get_h()));
      }
    }else{
      for(int ic=0; ic<8; ic++){
        Node &child = Node_heap[cfirst + ic];
        if(child.np > 0){
          child.make_boundary();
          bound_inner.merge(child.bound_inner);
          bound_outer.merge(child.bound_outer);
        }
      }
    }
  }
  
  void set_init_h(const float num, const float volume)
  {
    if(is_leaf()){
      const float roh = float(np) / volume;
      const float h0 = cbrtf(num / roh);
      for(int ip=0; ip<np; ip++){
        Particle &p = ptcl[ip+pfirst];
        p.set_h(h0);
      }
    }else{
      for(int ic=0; ic<8; ic++){
        Node &child = Node_heap[cfirst + ic];
        if(child.np > 0){
          child.set_init_h(num, volume / 8.0f);
        }
      }
    }
  }
  

  static void find_neib_beween_leaves(const Node &ileaf, const Node &jleaf)
  {
    for(int i=0; i<ileaf.np; i++){
      Particle &ip = ptcl[i+ileaf.pfirst];
      Boundary ibound(ip.pos, ip.get_h());
      if(not_overlapped(ibound, jleaf.bound_inner)) continue;
      float h2 = ip.get_h() * ip.get_h();
      if (ip.nnb < 2222)
        for(int j=0; j<jleaf.np; j++){
          Particle &jp = ptcl[j+jleaf.pfirst];
          const float r2 = (jp.pos - ip.pos).norm2();
          if(r2 < h2)
          {
            ip.nnb++;
            lInteract(ip, jp, r2);
          }
        }
    }
  }
  
  friend void operator << (Node &iNode, Node &jNode)
  {
    if(overlapped(iNode.bound_outer, jNode.bound_inner)){
      bool itravel = false;
      bool jtravel = false;
      if(iNode.is_leaf()){
        if(jNode.is_leaf()){
          find_neib_beween_leaves(iNode, jNode);
          return;
        }else{
          jtravel = true;
        }
      }else{
        if(jNode.is_leaf()){
          itravel = true;
        }else{
          if(iNode.depth < jNode.depth){
            itravel = true;
          }else{
            jtravel = true;
          }
        }
      }
      if(itravel){
        for(int i=0; i<8; i++){
          Node &ichild = Node_heap[i+iNode.cfirst];
          if(ichild.np == 0) continue;
          ichild << jNode;
        }
        return;
      }
      if(jtravel){
        for(int j=0; j<8; j++){
          Node &jchild = Node_heap[j+jNode.cfirst];
          if(jchild.np == 0) continue;
          iNode << jchild;
        }
        return;
      }
    }
  }
  
  void find_group_Node(
      int ncrit,
      std::vector<Node *> &group_list){
    if (np == 0)
      return;
    if(np < ncrit){
      group_list.push_back(this);
    }else{
      for(int ic=0; ic<8; ic++){
        Node &child = Node_heap[cfirst + ic];
        child.find_group_Node(ncrit, group_list);
      }
    }
  }
  
  friend void operator << (Particle &ip, Node &jNode)
  {
    Boundary bi(ip.pos, ip.get_h());
    if (ip.nnb < 2222)
      if(overlapped(bi, jNode.bound_inner)){
        if(jNode.is_leaf()){
          float h2 = ip.get_h() * ip.get_h();
          for(int j=0; j<jNode.np; j++)
          {
            Particle &jp = ptcl[j+jNode.pfirst];
            const float r2 = (jp.pos - ip.pos).norm2();
            if(r2 < h2)
            {
              /* do math here */
              ip.nnb++;
              lInteract(ip, jp, r2);
            }
          }
        }else{
          for(int j=0; j<8; j++){
            Node &jchild = Node_heap[j+jNode.cfirst];
            if(jchild.np == 0) continue;
            ip << jchild;
          }
        }
      }
  }

};
