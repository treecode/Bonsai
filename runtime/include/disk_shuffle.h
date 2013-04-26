struct DiskShuffle
{

  private:
    void read_file(
        FILE *in, 
        std::vector<star_particle> &sp, 
        std::vector<dark_particle> &dp,
        dump &h)
    {
      sp.clear();
      dp.clear();

      fread(&h, sizeof(h), 1, in);

      const float time = h.time;
      const int nstar = h.nstar;
      const int ndark = h.ndark;

      for(int i=0;i<ndark;i++)
      {
        dark_particle _dp;
        fread(&_dp, sizeof(dark_particle), 1, in);  
        dp.push_back(_dp);
      }
      for(int i=0;i<nstar;i++){     
        star_particle _sp;
        fread(&_sp, sizeof(star_particle), 1, in);
        sp.push_back(_sp);
      }
    }

    void rotate_xy(const float angl, float val[3]){
      float rot[3];
      rot[0] = cos(angl)*val[0] - sin(angl)*val[1];
      rot[1] = sin(angl)*val[0] + cos(angl)*val[1];
      rot[2] = val[2];

      val[0] = rot[0];
      val[1] = rot[1];
      val[2] = rot[2];
      //return rot;
    }

    void rotate_yz(const float angl, float val[3]){
      float rot[3];
      rot[1] = cos(angl)*val[1] - sin(angl)*val[2];
      rot[2] = sin(angl)*val[1] + cos(angl)*val[2];
      rot[0] = val[0];

      val[0] = rot[0];
      val[1] = rot[1];
      val[2] = rot[2];
      //return rot;
    }

    void rotate_one_particle(float pos[3], float vel[3])
    {
      const double angl = (2*drand48()-1.)*M_PI;
      rotate_xy(angl, pos);
      rotate_xy(angl, vel);
      const float f = 1.0e-3;
      pos[0] *= 1 + f * (2.0*drand48()- 1.0);
      pos[1] *= 1 + f * (2.0*drand48()- 1.0);
      pos[2] *= 1 + f * (2.0*drand48()- 1.0);
      vel[0] *= 1 + f * (2.0*drand48()- 1.0);
      vel[1] *= 1 + f * (2.0*drand48()- 1.0);
      vel[2] *= 1 + f * (2.0*drand48()- 1.0);
    }

    std::vector<dvec3> _pos, _vel;
    std::vector<double> _mass;
    int nstar, ndark;

  public:
    DiskShuffle(const std::string &fileName)
    {
      FILE *fin = fopen(fileName.c_str(), "rb");
      if (!fin)
      {
        fprintf(stderr, "DiskShuffle:: file %s not found\n", fileName.c_str());
        assert(0);
      }

      std::vector<star_particle> sp;
      std::vector<dark_particle> dp;
      dump h;
      read_file(fin, sp, dp, h);

      _pos.clear();
      _vel.clear();
      _mass.clear();

      nstar = h.nstar;
      ndark = h.ndark;

      const int n = h.nstar+h.ndark;
      _pos.reserve(n);
      _vel.reserve(n);
      _mass.reserve(n);

      for (int i = 0; i < h.nstar; i++)
      {
        rotate_one_particle(sp[i].pos, sp[i].vel);
        _pos.push_back(dvec3(sp[i].pos[0], sp[i].pos[1], sp[i].pos[2]));
        _vel.push_back(dvec3(sp[i].vel[0], sp[i].vel[1], sp[i].vel[2]));
        _mass.push_back(sp[i].mass);
      }
      for (int i = 0; i < h.ndark; i++)
      {
        rotate_one_particle(dp[i].pos, dp[i].vel);
        _pos.push_back(dvec3(dp[i].pos[0], dp[i].pos[1], dp[i].pos[2]));
        _vel.push_back(dvec3(dp[i].vel[0], dp[i].vel[1], dp[i].vel[2]));
        _mass.push_back(dp[i].mass);
      }
    }

    int get_nstar() const { return nstar; }
    int get_ndark() const { return ndark; }
    int get_ntot () const {return nstar + ndark;}
    const dvec3&  pos (const int i) const { return _pos[i]; }
    const dvec3&  vel (const int i) const { return _vel[i]; }
    const double& mass(const int i) const { return _mass[i]; }
};
