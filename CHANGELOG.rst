.. _changelog:

=========
CHANGELOG
=========


.. _changelog-v0.2.6:

v0.2.6 (2025-03-24)
===================

Bug Fixes
---------

* fix: add note in readme about JAX and cupy (`d3cd8d9`_)

Unknown
-------

* add runtimes (`6cdfb28`_)

* add runtime plot (`14b250b`_)

* add runtime doc (`3d8b4cd`_)

* add runtime doc (`7230f92`_)

* add more tests and runtime doc (`b8dce3f`_)

* add roundtrip comparisons (`81d6fc7`_)

* remove dt (`d78b4b3`_)

* remove mult (`b5ab983`_)

*  add corrected sinewave for roundtrip test, remove incorrect f0 test (`e2f8317`_)

* add WDM amplitude tests (`b99fc0d`_)

* add various transforms (`7d97420`_)

* fix cupy inverse (`b1f7dfa`_)

* add plotting (`b0ad4fa`_)

* add more tests (`aaeae9b`_)

* add more tests (`6627e03`_)

* add backend test (`ccb1193`_)

* add more logging (`f2d6541`_)

* add more logs (`0828947`_)

* add more tests (`8c80929`_)

* add more tests (`d1cb636`_)

* Merge branch 'main' of github.com:pywavelet/pywavelet (`ab6ba0e`_)

* Update __init__.py (`b607ae9`_)

* add more logs (`60e9525`_)

* add notes (`83e4ff6`_)

* increase the py version (`0d6568b`_)

* setup testing framework for cupy (`5009d79`_)

* add typing for phi (`16ef510`_)

* Check which backend (`f17fe1b`_)

* Check which backend (`196141c`_)

* add typing libs (beartype + Jaxtyping) (`3bffe6a`_)

* Add support for more backends (`dadf161`_)

* Adding cupy methods (`9fe44ab`_)

* Remove unnecessary beta(d,d,1) (`19b6d8e`_)

* Remove unused imports (`a2b42c8`_)

* run formatters (`380fba8`_)

* begin adding cupy (`93e0123`_)

* add more tests (`a91f509`_)

* Merge branch 'main' of github.com:avivajpeyi/pywavelet into main (`a149dc4`_)

* [pre-commit.ci] pre-commit autoupdate (#21)

* [pre-commit.ci] pre-commit autoupdate

updates:
- [github.com/pre-commit/pre-commit-hooks: v4.5.0 → v5.0.0](https://github.com/pre-commit/pre-commit-hooks/compare/v4.5.0...v5.0.0)
- https://github.com/pre-commit/mirrors-isort → https://github.com/PyCQA/isort
- [github.com/PyCQA/isort: v5.10.1 → 6.0.1](https://github.com/PyCQA/isort/compare/v5.10.1...6.0.1)
- https://github.com/ambv/black → https://github.com/psf/black
- [github.com/psf/black: 23.10.0 → 25.1.0](https://github.com/psf/black/compare/23.10.0...25.1.0)
- [github.com/psf/black: 23.10.0 → 25.1.0](https://github.com/psf/black/compare/23.10.0...25.1.0)

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com> (`6623f65`_)

* add check for vmin > vmax (`2d6eec3`_)

* more plotting for phi (`b124dc7`_)

* add additional JAX tests (`c0d8396`_)

.. _d3cd8d9: https://github.com/pywavelet/pywavelet/commit/d3cd8d92b5e6cf398ff3a4948b911533abe842f1
.. _6cdfb28: https://github.com/pywavelet/pywavelet/commit/6cdfb28b152a7f7ad499f0b6ba6ef69da9284c57
.. _14b250b: https://github.com/pywavelet/pywavelet/commit/14b250b55dbea88c3b7c22b5faca531113d34477
.. _3d8b4cd: https://github.com/pywavelet/pywavelet/commit/3d8b4cdc8dea6af33b08985cb97f0897984838fc
.. _7230f92: https://github.com/pywavelet/pywavelet/commit/7230f92dee98cea0e402111ad279f78c6134d565
.. _b8dce3f: https://github.com/pywavelet/pywavelet/commit/b8dce3fdf2f0b1261b6a74aa8af03465c87017ff
.. _81d6fc7: https://github.com/pywavelet/pywavelet/commit/81d6fc73ec9c37d7e5990f7008f1a468490b17ea
.. _d78b4b3: https://github.com/pywavelet/pywavelet/commit/d78b4b32b81b1b2e8bc6c4808c088df140f316dd
.. _b5ab983: https://github.com/pywavelet/pywavelet/commit/b5ab983f88a4dc5f6b7e75aa9c974f1c0c601d03
.. _e2f8317: https://github.com/pywavelet/pywavelet/commit/e2f8317f980927f34fc447e3b093ca43e2f8f3c2
.. _b99fc0d: https://github.com/pywavelet/pywavelet/commit/b99fc0d3a29b8c741c92e6a1cae9eabc409d1fbc
.. _7d97420: https://github.com/pywavelet/pywavelet/commit/7d97420b945dd8abb4dd9e201246719d39c4bc4c
.. _b1f7dfa: https://github.com/pywavelet/pywavelet/commit/b1f7dfa17df3215260150e41c0cf84e76a354d1b
.. _b0ad4fa: https://github.com/pywavelet/pywavelet/commit/b0ad4fad4537bf3375bb100f6b5be3042291e31e
.. _aaeae9b: https://github.com/pywavelet/pywavelet/commit/aaeae9b232e4328bfa2a566c7438f41f2dca0e31
.. _6627e03: https://github.com/pywavelet/pywavelet/commit/6627e0353d539b358e8bc6e786d3442a2b2b8072
.. _ccb1193: https://github.com/pywavelet/pywavelet/commit/ccb1193e968c9617ed866318326421fb2ae80645
.. _f2d6541: https://github.com/pywavelet/pywavelet/commit/f2d6541caa41f171a19abd5cfdb3e132fc831b8b
.. _0828947: https://github.com/pywavelet/pywavelet/commit/082894710eaa2f4a485a86524917068e3098b0c4
.. _8c80929: https://github.com/pywavelet/pywavelet/commit/8c80929e35211d482d15845fd9ef0cca62704b4b
.. _d1cb636: https://github.com/pywavelet/pywavelet/commit/d1cb636fb92fbec238de89129729a70f7fa33bb6
.. _ab6ba0e: https://github.com/pywavelet/pywavelet/commit/ab6ba0ed98a6597c9c8aef524a2f03b1b6fbd13e
.. _b607ae9: https://github.com/pywavelet/pywavelet/commit/b607ae971f1b04870f41bbaee95a2a30eef628d4
.. _60e9525: https://github.com/pywavelet/pywavelet/commit/60e952536a2812f474a2845f81dee52884b1685b
.. _83e4ff6: https://github.com/pywavelet/pywavelet/commit/83e4ff6e5ac5ee1fd51970a73c0401106cff453a
.. _0d6568b: https://github.com/pywavelet/pywavelet/commit/0d6568bc305d85ac7f5733365e50e51538d45329
.. _5009d79: https://github.com/pywavelet/pywavelet/commit/5009d7957d9d796a9e2258659e2815a35354a7a5
.. _16ef510: https://github.com/pywavelet/pywavelet/commit/16ef51086b1ba10db0d4f252caf3a560fee4f06b
.. _f17fe1b: https://github.com/pywavelet/pywavelet/commit/f17fe1b0cddb4376bdb18a092d6f83bc48dd9498
.. _196141c: https://github.com/pywavelet/pywavelet/commit/196141ce52946228a92aa1f4b0c14fdc66cb44a1
.. _3bffe6a: https://github.com/pywavelet/pywavelet/commit/3bffe6a7af4000b1bcece2e2fb1c56487980dd15
.. _dadf161: https://github.com/pywavelet/pywavelet/commit/dadf161bed6c049c8c24716f475f94b9d6e9fabf
.. _9fe44ab: https://github.com/pywavelet/pywavelet/commit/9fe44ab0d0862110137e02d746e1af1591373c66
.. _19b6d8e: https://github.com/pywavelet/pywavelet/commit/19b6d8e169c87f780afe809863762574a9a6334b
.. _a2b42c8: https://github.com/pywavelet/pywavelet/commit/a2b42c885dd248a2787b6ce7ae549f3d15044af3
.. _380fba8: https://github.com/pywavelet/pywavelet/commit/380fba8f074d7fbe9bc4ed51461688c45dfc7b0a
.. _93e0123: https://github.com/pywavelet/pywavelet/commit/93e0123dce8f4bf44249489a55a84865ef989592
.. _a91f509: https://github.com/pywavelet/pywavelet/commit/a91f509c1c7e18d75c6ceeff22242b1e463f4191
.. _a149dc4: https://github.com/pywavelet/pywavelet/commit/a149dc4178581f26745b0725346ad469f4aed8d6
.. _6623f65: https://github.com/pywavelet/pywavelet/commit/6623f65aa13a020120d52363c2bb7b6dd425845f
.. _2d6eec3: https://github.com/pywavelet/pywavelet/commit/2d6eec342676c1dadb5ce7318edf781572dde5c4
.. _b124dc7: https://github.com/pywavelet/pywavelet/commit/b124dc75c13540a946b665e0bbb70de43ffe16fd
.. _c0d8396: https://github.com/pywavelet/pywavelet/commit/c0d8396587737ac9d9559e79c175a043273b8ce6


.. _changelog-v0.2.5:

v0.2.5 (2025-01-28)
===================

Bug Fixes
---------

* fix: filter first then window (#20)

* fix: filter first then window

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com> (`3228f7b`_)

Chores
------

* chore(release): 0.2.5 (`34b18c7`_)

.. _3228f7b: https://github.com/pywavelet/pywavelet/commit/3228f7be0d7efb48812920e822a14e795ebac57f
.. _34b18c7: https://github.com/pywavelet/pywavelet/commit/34b18c7d4074992d6c1ce75806efb3edab5ce49b


.. _changelog-v0.2.4:

v0.2.4 (2025-01-24)
===================

Chores
------

* chore(release): 0.2.4 (`8ea5b2f`_)

Unknown
-------

* Merge branch 'main' of github.com:avivajpeyi/pywavelet into main (`d2c84d9`_)

.. _8ea5b2f: https://github.com/pywavelet/pywavelet/commit/8ea5b2fc0ffa5a1cb274273ec48e164f8bd2064a
.. _d2c84d9: https://github.com/pywavelet/pywavelet/commit/d2c84d980b1701baf99e40ba6191cbd9336cfa59


.. _changelog-v0.2.3:

v0.2.3 (2025-01-24)
===================

Bug Fixes
---------

* fix: add masks to filter out gaps (`26fe40a`_)

* fix: add backend check for os.environ (`98c0818`_)

* fix: add test for jax (`1940394`_)

Chores
------

* chore(release): 0.2.3 (`d067461`_)

.. _26fe40a: https://github.com/pywavelet/pywavelet/commit/26fe40ace80d5f9d598e1efeba2f8ca4a6f1043b
.. _98c0818: https://github.com/pywavelet/pywavelet/commit/98c0818078190d829a23734f932f1f93c9932167
.. _1940394: https://github.com/pywavelet/pywavelet/commit/194039437a3a9b3ada303d101b4e2573ab7d0afd
.. _d067461: https://github.com/pywavelet/pywavelet/commit/d0674615df328774a0d80eb224b5c503fbd8f332


.. _changelog-v0.2.2:

v0.2.2 (2025-01-23)
===================

Chores
------

* chore(release): 0.2.2 (`eed5d68`_)

Unknown
-------

* Merge branch 'main' of github.com:pywavelet/pywavelet (`e8e2115`_)

.. _eed5d68: https://github.com/pywavelet/pywavelet/commit/eed5d6864276fc5f90c4866749903e3e358df5ca
.. _e8e2115: https://github.com/pywavelet/pywavelet/commit/e8e2115e797a5001f236ff027a14ef226151dcc1


.. _changelog-v0.2.1:

v0.2.1 (2025-01-23)
===================

Bug Fixes
---------

* fix: fix readme path (`34e927d`_)

Chores
------

* chore(release): 0.2.1 (`a55bce5`_)

Unknown
-------

* Merge branch 'main' of github.com:pywavelet/pywavelet (`b8c6d15`_)

.. _34e927d: https://github.com/pywavelet/pywavelet/commit/34e927d411ec8fde89f552bd5ec89b38820e07e0
.. _a55bce5: https://github.com/pywavelet/pywavelet/commit/a55bce518c3484543efada283399a41df3ecf001
.. _b8c6d15: https://github.com/pywavelet/pywavelet/commit/b8c6d1579d48ec5fa22130430267794ae8e54f6c


.. _changelog-v0.2.0:

v0.2.0 (2025-01-23)
===================

Bug Fixes
---------

* fix: add temp inverse transform (`586f9ad`_)

Chores
------

* chore(release): 0.2.0 (`fcc35e9`_)

Features
--------

* feat: add jax as optional backend (`264613e`_)

.. _586f9ad: https://github.com/pywavelet/pywavelet/commit/586f9ad311f905f7d2fbbfd02fea8198eeda8237
.. _fcc35e9: https://github.com/pywavelet/pywavelet/commit/fcc35e973d906bd18e03204449564f35fc657b89
.. _264613e: https://github.com/pywavelet/pywavelet/commit/264613e5a58042641eb6814530dab36bb54b3371


.. _changelog-v0.1.2:

v0.1.2 (2025-01-20)
===================

Bug Fixes
---------

* fix: adjust the WaveletMask repr (`50c6201`_)

Chores
------

* chore(release): 0.1.2 (`7b64515`_)

.. _50c6201: https://github.com/pywavelet/pywavelet/commit/50c6201efb7689dd9757a5e4c6047d241015cb96
.. _7b64515: https://github.com/pywavelet/pywavelet/commit/7b64515a2bf2418719f68cb6b15f1c204938408d


.. _changelog-v0.1.1:

v0.1.1 (2025-01-16)
===================

Chores
------

* chore(release): 0.1.1 (`bd15453`_)

Unknown
-------

* Merge branch 'main' of github.com:avivajpeyi/pywavelet into main (`69eefa2`_)

.. _bd15453: https://github.com/pywavelet/pywavelet/commit/bd15453e028705548232b802b2d21bbebd307ca7
.. _69eefa2: https://github.com/pywavelet/pywavelet/commit/69eefa29b7873c30fcb74ad1e051eb20101a277a


.. _changelog-v0.1.0:

v0.1.0 (2025-01-15)
===================

Bug Fixes
---------

* fix: refactor type outside transforms (`efb8878`_)

Chores
------

* chore(release): 0.1.0 (`c5a3fde`_)

Features
--------

* feat: add wavelet mask and more tests (`e009903`_)

.. _efb8878: https://github.com/pywavelet/pywavelet/commit/efb88789f8468ff18f99abaf6168bb8fc0f5947b
.. _c5a3fde: https://github.com/pywavelet/pywavelet/commit/c5a3fdea455c16478f04049f14bc35dfcf4efb15
.. _e009903: https://github.com/pywavelet/pywavelet/commit/e00990300d9c013438580c2bc47ea93570fd95be


.. _changelog-v0.0.5:

v0.0.5 (2024-12-12)
===================

Bug Fixes
---------

* fix: update changelog generator (`884c87b`_)

Chores
------

* chore(release): 0.0.5 (`4ed6b03`_)

.. _884c87b: https://github.com/pywavelet/pywavelet/commit/884c87bcd36b5d21eb1a8e10ee9e0edf6f65d744
.. _4ed6b03: https://github.com/pywavelet/pywavelet/commit/4ed6b03618347cc179195feec57b05e04a004100


.. _changelog-v0.0.4:

v0.0.4 (2024-12-12)
===================

Chores
------

* chore(release): 0.0.4 (`fad0b89`_)

Unknown
-------

* Merge branch 'main' of github.com:avivajpeyi/pywavelet into main (`4c04fb4`_)

.. _fad0b89: https://github.com/pywavelet/pywavelet/commit/fad0b8913d7160ca498938e67131b8006ff65580
.. _4c04fb4: https://github.com/pywavelet/pywavelet/commit/4c04fb4a4dc39bce8617dfe98d405ad803fd8657


.. _changelog-v0.0.3:

v0.0.3 (2024-12-12)
===================

Bug Fixes
---------

* fix: adjust changelog (`4abd6a7`_)

* fix: update action versions (`8f78223`_)

Chores
------

* chore(release): 0.0.3 (`eb5d545`_)

Unknown
-------

* Merge branch 'main' of github.com:avivajpeyi/pywavelet into main (`1515575`_)

* improve pltting label (`6e41f67`_)

* improve repr (`fc17731`_)

* plotting fix for log scale wavelet (`c3e2819`_)

* fix SNR test (`a3e8878`_)

* fix filtering (`038f674`_)

* add impoved repr (`dbec3e1`_)

* reorder improts (`a7309ce`_)

* add QOL fixes, __repr__, slicing (`9468d01`_)

* fix unnits (`2fd64f5`_)

* add plotting fix (`afc62ee`_)

* add filtering options (`9eea8e1`_)

* fix sshape bug (`9e2bcc9`_)

* test freq.to_wavelet, wavelet.to_freq covertors, SNR computation (`9ccc8ab`_)

* add to_wavelet, to_freqseries, "==" operator (`7fe72d3`_)

* add inner-product, snr computation (`f6a20ed`_)

* plotting fixes (`0608841`_)

* add plot with nans and wavelet-trend plo (`b2a92cb`_)

* add "==" (`27f8475`_)

* add '+' and '-' operations (`1c9450c`_)

* add nan-color option (`c69adee`_)

* add + and - operations for wavelet (`1754fcf`_)

* Merge branch 'main' of github.com:pywavelet/pywavelet (`92d8497`_)

* add plotting flag for Nan matrix (`ab74b42`_)

* patch: update release method (`eab9964`_)

* update ignore (`31b21b0`_)

* move FFT component to test out rocket-fftt (`eb75ca5`_)

* jit functions (`0ec3f5f`_)

* fix email (`29c3634`_)

* improve docstrings (`3753b23`_)

* add time-formatters (`ae09987`_)

* add t0 and improve repr (`bb44e70`_)

* add t0 and improve repr (`27aedc8`_)

* use     t_bins+= data.t0 instead of data.time[0] (`2713bd8`_)

* add likelihood (`97bfba1`_)

* remove unused packages and clean up datatype (`4df4ab2`_)

* replace loguru with rich (`5e6cdc1`_)

.. _4abd6a7: https://github.com/pywavelet/pywavelet/commit/4abd6a70b3c563d597f312552f4e37a0f8e3e3d4
.. _8f78223: https://github.com/pywavelet/pywavelet/commit/8f782233f30c663e50c8c972773d3ab72807f34f
.. _eb5d545: https://github.com/pywavelet/pywavelet/commit/eb5d545243ef247c74fe49f0e8253d86ae627013
.. _1515575: https://github.com/pywavelet/pywavelet/commit/1515575513c82290e28923ba7c51cfff98a10341
.. _6e41f67: https://github.com/pywavelet/pywavelet/commit/6e41f67da855754d97ee687cd22a930c07a6433e
.. _fc17731: https://github.com/pywavelet/pywavelet/commit/fc17731e4f542c942774c19d63f5c962dfcbe3ac
.. _c3e2819: https://github.com/pywavelet/pywavelet/commit/c3e2819f54a4ffc3141d3e67961dbcdcafa5b0c4
.. _a3e8878: https://github.com/pywavelet/pywavelet/commit/a3e88788f289309678e9c03a33f08ef10b087a0f
.. _038f674: https://github.com/pywavelet/pywavelet/commit/038f6742c89ca75da1e4cebfde70ae00a4d8fa76
.. _dbec3e1: https://github.com/pywavelet/pywavelet/commit/dbec3e1f491b6c3d66c04ca609b218cf31197acf
.. _a7309ce: https://github.com/pywavelet/pywavelet/commit/a7309ce7be7170bdf580df79ac2dddd438c61611
.. _9468d01: https://github.com/pywavelet/pywavelet/commit/9468d0197756fe220eb38a2cf68041b238177b49
.. _2fd64f5: https://github.com/pywavelet/pywavelet/commit/2fd64f503a857bcdf1a40b672a8ba93fc2663321
.. _afc62ee: https://github.com/pywavelet/pywavelet/commit/afc62ee51902138f06f1b23c367187c689760e2e
.. _9eea8e1: https://github.com/pywavelet/pywavelet/commit/9eea8e1be152d9174721826e04a4983fcf374896
.. _9e2bcc9: https://github.com/pywavelet/pywavelet/commit/9e2bcc9d0d14d3c4f4b7131c589f80084bf65ce8
.. _9ccc8ab: https://github.com/pywavelet/pywavelet/commit/9ccc8ab24a34f09b6f8daef98909b3c5d8d65057
.. _7fe72d3: https://github.com/pywavelet/pywavelet/commit/7fe72d3d166cdd30813094c2e5db30a16dcbb614
.. _f6a20ed: https://github.com/pywavelet/pywavelet/commit/f6a20ed6b3d23fa81293354527ea71e15fdba4a0
.. _0608841: https://github.com/pywavelet/pywavelet/commit/060884127ba8c9bc76f1066962f047c51dee65f6
.. _b2a92cb: https://github.com/pywavelet/pywavelet/commit/b2a92cbcb32445fdd44321ea11b9c9ffe0168d3d
.. _27f8475: https://github.com/pywavelet/pywavelet/commit/27f847537409f468d9143799f5992064dbc36bbd
.. _1c9450c: https://github.com/pywavelet/pywavelet/commit/1c9450c112c6a5449fd1b46b5af383ea60e34b8c
.. _c69adee: https://github.com/pywavelet/pywavelet/commit/c69adee82801c8a027f7d5d352f8dac0fefbda72
.. _1754fcf: https://github.com/pywavelet/pywavelet/commit/1754fcf08f095788f2c3e639931a4a75db4795ef
.. _92d8497: https://github.com/pywavelet/pywavelet/commit/92d8497f5f6f2724b0a5bde75633e314b32d01ea
.. _ab74b42: https://github.com/pywavelet/pywavelet/commit/ab74b42a583e4782fd9b67ae2b2e61be13d7f93b
.. _eab9964: https://github.com/pywavelet/pywavelet/commit/eab9964e0332262d337d2df40f327a9970b715c7
.. _31b21b0: https://github.com/pywavelet/pywavelet/commit/31b21b07bffa9f12ea1f205ae0d20b8165465e5f
.. _eb75ca5: https://github.com/pywavelet/pywavelet/commit/eb75ca5c7ab2e71ce8cd14b8abce850bf5fef450
.. _0ec3f5f: https://github.com/pywavelet/pywavelet/commit/0ec3f5f8258d523d0a290f65315afd10ee9662d7
.. _29c3634: https://github.com/pywavelet/pywavelet/commit/29c3634d71bb21925af4b53c466789f0a6336fad
.. _3753b23: https://github.com/pywavelet/pywavelet/commit/3753b23741fb88f5a1ee02804971b00ec5cd9e97
.. _ae09987: https://github.com/pywavelet/pywavelet/commit/ae0998737d44251a87b100d3d6af5337eab9ee0f
.. _bb44e70: https://github.com/pywavelet/pywavelet/commit/bb44e70475ea44d297ce6a286a4d24b7111aead7
.. _27aedc8: https://github.com/pywavelet/pywavelet/commit/27aedc836853c08523c3c6225ada1a3da42dcde6
.. _2713bd8: https://github.com/pywavelet/pywavelet/commit/2713bd840f4efb1644db101602392cc68a57b3c3
.. _97bfba1: https://github.com/pywavelet/pywavelet/commit/97bfba128523c1469625f6047867d490bd231f51
.. _4df4ab2: https://github.com/pywavelet/pywavelet/commit/4df4ab295a7fae48f18d99e7ea065d3786f989f5
.. _5e6cdc1: https://github.com/pywavelet/pywavelet/commit/5e6cdc1cf6b26ad652598fc6be1a27a5e077a905


.. _changelog-v0.0.2:

v0.0.2 (2024-10-15)
===================

Unknown
-------

* v0.0.2 (`789ed95`_)

* fix docs (`8114ed2`_)

* fix tests (`42f3f4b`_)

* remove unnused (`fa4383f`_)

* fix transform (`11f435e`_)

* fix datasets (`32ea95c`_)

* add tests for freq-time domiain types, fix SNR monochromatic signal check (`e0e018c`_)

* hacking on SNR and analytical example (`cf1e441`_)

* fix SNR (`cd1e8d9`_)

* fix snr test (`8b1f232`_)

* add hacks with giorgio and ollie on sinewave testing (`20f376a`_)

* axis label (`619b55f`_)

* update log (`eabd019`_)

* refactor tests (`5d42f6b`_)

* add cbar label (`7303cdf`_)

* plotting fixes (`dd48d64`_)

* add direct WDM comparison (`f2c82a6`_)

* add branch check (`8ff9493`_)

* add branch to plot dir (`84c566e`_)

* remove unused imports (`4ee06a0`_)

* fix test (`8ccad52`_)

* Merge pull request #18 from pywavelet/get_rid_of_datatype_class_in_prep_for_jax

cleaup [prep fr jax] (`704e9c1`_)

* pytest fixes (`8697db0`_)

* all tests passing (`dc3e02f`_)

* time->wdm->time passing (`48724e1`_)

* cleaup (`2d06b46`_)

* add docs (`5ddbdc8`_)

* add roundtrip exmple (`f5976fd`_)

* fix twosied error (`7798720`_)

* fix docs (`db73d7a`_)

* refactor plotting (`65350de`_)

* typing hint fixes (`5982405`_)

* refactor dataobj (`63151a4`_)

* cleanup (`365d89a`_)

* clean up docs (`dfe3136`_)

* disable JIT for now (`5cf5f80`_)

* plot abs(residuals) (`8d87d72`_)

* refactor docs (`0de37c8`_)

* remove unused tests and consolidate (`d777222`_)

* remove CBC waveform (`fdaf7d9`_)

* Add wavelet-plotting (`fc25966`_)

* Remove LVK + LISA examples (will be in separate case studies) (`995871e`_)

* consolidate utils to evol-psd and compute_snr (`5e59153`_)

* clean up PSD test to only test evol-psd (`7976d20`_)

* move evolutionary-PSD to utils (`e6d88cd`_)

* Merge branch 'main' of github.com:pywavelet/pywavelet (`cac0da9`_)

* Update README.rst (`7093025`_)

* Update README.rst

tidied up readme for others to install (`8ea7436`_)

* remove GW170817 example (`bd55639`_)

* remove examples test (`3f763fb`_)

* fix version test (`e676e65`_)

* Merge pull request #12 from pywavelet/allow_precommit_fail

allow precommit failure (`efc5b1f`_)

* allow precommit failure (`0410893`_)

* delete waveform-generator test (`02d984d`_)

* add test (`980875b`_)

* fix formatting (`673f33c`_)

* remove wavelet_dataset (`c8c8f37`_)

* turn off CBC waveform generator (`727c47d`_)

* add logo (`7893845`_)

* Merge branch 'refactoring' (`985e9eb`_)

* add deprecation warning for ollie (`1ee69b4`_)

* rename Data->CoupledData (`dae0fb0`_)

* Merge pull request #11 from pywavelet/refactoring

refactoring: removing unsued files, moving functions around, running linter (`077e58e`_)

* removing unsued files, moving functions around, running linter (`fd88319`_)

* readying for merge (`8552f77`_)

* added in error checking for boundary (`3ee0be1`_)

* investigating non-monochromatic signals (`4411c74`_)

* added kwargs for plots, title (`0a05d8d`_)

* removed LISA example (`68bf006`_)

* fixed small bug (`53e1768`_)

* functions now jitted for speed (`fd7628e`_)

* tidied up, deleted pieces (`ca545d4`_)

* fixed bug in phi. B = dOmega - 2*A (`77666f9`_)

* Merge branch 'main' of https://github.com/avivajpeyi/pywavelet (`6253208`_)

* Merge pull request #7 from pywavelet/inverse_transforms

Inverse transforms (`6ff8501`_)

* removed bilby + pycbc (`7b58b43`_)

* removed breakpoints (`f73a3dc`_)

* removed importing bilby + pycbc (`547fd32`_)

* tidied up, removed uneccessary variables (`5e5f2e1`_)

* removed irritating breakpoints, sorry (`be9778f`_)

* added time domain inverse checks (`6d704c0`_)

* correct normalisation, mult by (2/Nf) (`a4083f4`_)

* correct normalisation now (`c77e2fe`_)

* Fixed normalisation

I was trying to be clever and include Nf/2 into the window function here.

This is not the correct noramlisation and this screwed the inverse transform up. I have
placed it in front of the wavelet transform instead. This I believe is correct (`a9a0610`_)

* corrected dimensions, backwards transform works now (`a811f24`_)

* added numba to inv funcs (`a0424ef`_)

* Fixed inverse transform (wavelet -> freq)

The dimensions were screwed up (N_t <-> N_f).

I added the lazy solution, just taking a transpose of the wavelet
coefficient matrix. This has worked. I've also included the correct
normalising constants so that it agrees with the usual FFT.

Everything is consistent now. (`a1cb77b`_)

* changed mult to 16 (`2a1f889`_)

* removed mask, fixed length (`9d6b379`_)

* removed N = len(data) bug (`da3d090`_)

* removed tukey function (`822d19b`_)

* Normalising constants, understood.

Matt's code is different from Cornish's code. For Matt's code to be consistent with our
frequency domain code, we require a normalising factor in front of the Meyer window
of the form $(Nf/2) \cdot \pi^{-1/2}$. On this specific commit, there are a load of
comments in the function phitilde_vec_norm indicating parts we need to understand.

The nice thing though is that analytically, for monochromatic signals, we now
have an expression for the wavelet coefficients $\omega_{nm} = A\sqrt{2N_{f}}$ for
n odd and m even. With the conventions above, we have verified this + checked the SNRs.

I'm now happy with this transformation code. (`501fae1`_)

* removed case studies into own repository (`7c5f347`_)

* fixed bug in residuals (`257f43d`_)

* using proper monochromatic sinusoid (`20a421d`_)

* from_wavelet_to_freq, freqs now positive (`c3b9438`_)

* changed PSD to periodigram, title (`206a5d7`_)

* fixed bug in length N (`15949df`_)

* analytical formula monochrom signal (`a04a76e`_)

* Merge pull request #6 from pywavelet/roundtrip_hacking

Roundtrip hacking (`6572581`_)

* work through NDs (`decfe7f`_)

* fix plotting issue (`6209923`_)

* var renaminng (`49cb11c`_)

* merge into one function (`fda592d`_)

* add roundtrip from t->wdm->t, t->f->wdm->f (`2f6810e`_)

* Add notes to why we cant merge this into one function (`028349e`_)

* [black] (`7cd06af`_)

* test_basic, changed dt (`7f4ece1`_)

* start fixing psd errors (`ffea941`_)

* Merge remote-tracking branch 'origin/main' (`36a7279`_)

* Merge branch 'main' of github.com:avivajpeyi/pywavelet (`dae3912`_)

* fix precommit (`9c109d8`_)

* bug found in generate_noise_from_psd, ndim (`92c20fe`_)

* fixed bug in test (`e15d5d3`_)

* all SNR tests working (`c8e651c`_)

* working with positive transform (`0d00f58`_)

* added sqrt(2)/dt into bilby waveform (`5927230`_)

* now using positive transform (`141cfac`_)

* now using positive transform (`b709b9e`_)

* testing, new commit, no change (`67948ed`_)

* comments (`d50e6e8`_)

* reorganised, no real changes (`7141a73`_)

* added script to try inverse transforms (`01c6050`_)

* extra factor sqrt(2) (`20f5d30`_)

* save plots (`0cdf9c1`_)

* remove breakpoints (`15e24f0`_)

* add pastamakers (`d65d993`_)

* remove pasta (`9024797`_)

* run precommit (`a04112e`_)

* extra comments (`4024ae6`_)

* few extra comments (`78a1f73`_)

* factor of sqrt(2) added in transform

Added in a factor of sqrt(2) to make sure that the SNRs agree. (`bc50c43`_)

* Changed FFT and fourier freqs

Ignoring windowing in the time domain. Also I am now setting freq[0] = freq[1] rather than
removing the 0th frequency bin from the DTFT. This will cause issues with the inverse transform.

setting freq[0] = freq[1] is fine since we only use this in the PSD. PSD[f = 0] = \infty so we want to
avoid using this. (`68b3eec`_)

* new file, checking inverse transforms (`0cec53c`_)

* Fixed bug for wavelet time bins

Before we were setting N = length of data, regardless of whether it is time or frequency domain.
This is only correct if we use a two-sided transform where the length of FFT = length in time domain.

For zero_padded signals (as they all should be, for speed), the rfft returns N/2 points. Hence, in order to get the
correct time bins, we need to double the data points if we take in a frequency series.

This was fine for the time domain, but incorrect for the frequency domain.

Ollie (`29665f5`_)

* Merge branch 'main' of https://github.com/avivajpeyi/pywavelet (`5dd5e0f`_)

* add snr (`416c810`_)

* ignoring .npy files gitignore (`d07ae7e`_)

* conventions sorted, delta_t dealt with (`113251b`_)

* conventions sorted, delta_t dealt with (`0c1820d`_)

* samples added (`e2b3767`_)

* working PE code, wavelets (`b1947f0`_)

* minor changes (`1596bde`_)

* analytical formulas, FFT (`0ca80ee`_)

* fix lnl (`35d2ce2`_)

* dt fix (round 1) (`ad43d13`_)

* dt hacking with ollie (`b2db4b3`_)

* pre-commit files (`3fbbaf4`_)

* Merge branch 'main' of github.com:avivajpeyi/pywavelet (`d13f219`_)

* add more tests -- hacking with Georgio (`7add237`_)

* add more tests -- hacking with Georgio (`f9fc53b`_)

* fix SNR (`5a5dff2`_)

* add SNR tests (`df6016e`_)

* add tests (`13d7dce`_)

* Merge branch 'main' of github.com:avivajpeyi/pywavelet (`31770ec`_)

* added snr test (`e50827b`_)

* add psd for lvk (`12776a4`_)

* add tests (`379bad7`_)

* hacking on snr (`73e9d42`_)

* add psd (`1f542bc`_)

* add utils (`631ab0c`_)

* add transform tests (`7b88b52`_)

* Merge remote-tracking branch 'origin/main'

# Conflicts:
#	src/pywavelet/psd.py
#	src/pywavelet/transforms/types.py
#	src/pywavelet/utils/lisa.py
#	src/pywavelet/utils/snr.py
#	src/pywavelet/utils/wavelet_bins.py
#	tests/test_psd.py
#	tests/test_roundtrip_conversion.py
#	tests/test_snr.py (`750b709`_)

* add titles (`64c12c9`_)

* precommit fixes (`70e6362`_)

* add quentin PSD (`e664c48`_)

* Merge branch 'main' of github.com:avivajpeyi/pywavelet into main (`39ce268`_)

* Add noise demo (`32a3998`_)

* add more transforms (`fe01f91`_)

* add psd test (`3f5b34c`_)

* add snr fix (`4c864e2`_)

* fix transposed matrix bug (`39f7526`_)

* add tests (`bbe764f`_)

* add PSD (`56664c3`_)

* added stationary PSD (`c1f4f92`_)

* add time and freq bins (`c62bcde`_)

* add nb black formatter (`1ca831c`_)

* hacking on xarray (`30d8444`_)

* fix meta data (`f502346`_)

* temp disable snr test (`f204ad1`_)

* remove dev install (`5c2b2f4`_)

* add write permission (`62d2fd6`_)

* update release action (`d04c1e4`_)

* refactor setup --> pyproject (`26ba587`_)

* add snr hacking (`0321216`_)

* add SNR (`fa5dab0`_)

* add LnL notes (`a42daaf`_)

* refactor (`9213db2`_)

* Simplify code (`45c6aa3`_)

* add plots for CBC wavelet transforms (`2d64fbe`_)

* Add CBC example (`e495a59`_)

* add waveform-generator template (`f37b03e`_)

* add waveform-generator template (`189c510`_)

* update docs (`79f4e0e`_)

* refactor code (`37869e6`_)

* added README (`97a0402`_)

* added basic MCMC code (`bfd3a13`_)

* init (`39119b4`_)

* first commit (`02fcc81`_)

.. _789ed95: https://github.com/pywavelet/pywavelet/commit/789ed9594a724c7884caa76cb8072cb0f5fe9187
.. _8114ed2: https://github.com/pywavelet/pywavelet/commit/8114ed221f44f8bc43ee587cd4b036ea9f3433f5
.. _42f3f4b: https://github.com/pywavelet/pywavelet/commit/42f3f4bfadf057824b9c03889653e8e81d9bba8f
.. _fa4383f: https://github.com/pywavelet/pywavelet/commit/fa4383f92d6e78630ddab40f6490e1368bd83444
.. _11f435e: https://github.com/pywavelet/pywavelet/commit/11f435e54f01117c8c0d2e12f9ee73567ed49687
.. _32ea95c: https://github.com/pywavelet/pywavelet/commit/32ea95c517c1f99d60aafe36ea8cbccccbfce114
.. _e0e018c: https://github.com/pywavelet/pywavelet/commit/e0e018cb63265302e640902b57802a9da34a0a28
.. _cf1e441: https://github.com/pywavelet/pywavelet/commit/cf1e44187380ebd94926cd708b10ae3cce40e10b
.. _cd1e8d9: https://github.com/pywavelet/pywavelet/commit/cd1e8d9cc49394f20ab85576489016a4bc832a9f
.. _8b1f232: https://github.com/pywavelet/pywavelet/commit/8b1f232f6df8956e70c871c270935ef4c0614585
.. _20f376a: https://github.com/pywavelet/pywavelet/commit/20f376a9e3a35e9858fee93b2ca41e5ed59c88af
.. _619b55f: https://github.com/pywavelet/pywavelet/commit/619b55f5c48d880703433b10caab4492debbd256
.. _eabd019: https://github.com/pywavelet/pywavelet/commit/eabd01942b214cb4ee1752dfdc6d17acbeb8be8c
.. _5d42f6b: https://github.com/pywavelet/pywavelet/commit/5d42f6b7f23cd3042fa4c6d56edd836fbb05b3d2
.. _7303cdf: https://github.com/pywavelet/pywavelet/commit/7303cdfda3d6405bfc8d218363beb2e687430d6e
.. _dd48d64: https://github.com/pywavelet/pywavelet/commit/dd48d64a2e3fe022461aaefd025f16433a3c37e3
.. _f2c82a6: https://github.com/pywavelet/pywavelet/commit/f2c82a6b6a904ff2edc7f5dddd0eaca0c71778c5
.. _8ff9493: https://github.com/pywavelet/pywavelet/commit/8ff9493c45ce6bd28a50a87279a84c8f8d423a3d
.. _84c566e: https://github.com/pywavelet/pywavelet/commit/84c566ebb3a3b9fab2f311a438772e1b35c6b9d9
.. _4ee06a0: https://github.com/pywavelet/pywavelet/commit/4ee06a02047fe6025ce9bc4965064808b2868556
.. _8ccad52: https://github.com/pywavelet/pywavelet/commit/8ccad52b023bdd9ed69f9a2ddc3a554bbd90e3f9
.. _704e9c1: https://github.com/pywavelet/pywavelet/commit/704e9c1c37513304fefa2a7848208ed5ee8cfd74
.. _8697db0: https://github.com/pywavelet/pywavelet/commit/8697db0dcee36648c7d4b8062ae57b8d56cb344f
.. _dc3e02f: https://github.com/pywavelet/pywavelet/commit/dc3e02fd48f4df87d5e2a16fdd7faf7e95d9cfe7
.. _48724e1: https://github.com/pywavelet/pywavelet/commit/48724e1714e812ab1593fb54a94da7f599f01d6b
.. _2d06b46: https://github.com/pywavelet/pywavelet/commit/2d06b46e492ddd816b66c4a55eff720e895254e2
.. _5ddbdc8: https://github.com/pywavelet/pywavelet/commit/5ddbdc88f52b1bea6f2414adfc0021a3723acce0
.. _f5976fd: https://github.com/pywavelet/pywavelet/commit/f5976fd65b1c68e36c248752d077aa11ca92b288
.. _7798720: https://github.com/pywavelet/pywavelet/commit/7798720ba0912f876f750bc24b21611dedb0dacf
.. _db73d7a: https://github.com/pywavelet/pywavelet/commit/db73d7a04fa84ea01cac863a08026d6ce5557d12
.. _65350de: https://github.com/pywavelet/pywavelet/commit/65350de3943bb2f6e95669b761b031c68ede28f8
.. _5982405: https://github.com/pywavelet/pywavelet/commit/5982405bafa07e4dbe040b7857c719137853805e
.. _63151a4: https://github.com/pywavelet/pywavelet/commit/63151a47cde9edc14f1e7e0bf17f554e78ad257c
.. _365d89a: https://github.com/pywavelet/pywavelet/commit/365d89a089289ebfea89979a656ff8a50fb851d2
.. _dfe3136: https://github.com/pywavelet/pywavelet/commit/dfe31363473f7a4f2f3b08ba5ca3506a5758d0a9
.. _5cf5f80: https://github.com/pywavelet/pywavelet/commit/5cf5f804a368438fdf38ac77d45f94705a5021e0
.. _8d87d72: https://github.com/pywavelet/pywavelet/commit/8d87d720ed84c1879a595d57926db17dbae1bd4c
.. _0de37c8: https://github.com/pywavelet/pywavelet/commit/0de37c8d850a5c595e6ed15dd5d02c0aa1c028cc
.. _d777222: https://github.com/pywavelet/pywavelet/commit/d77722289a87f475ee660163e6f9adb50acac994
.. _fdaf7d9: https://github.com/pywavelet/pywavelet/commit/fdaf7d9ad6e2abe16bfd820cbea380dca9cb021b
.. _fc25966: https://github.com/pywavelet/pywavelet/commit/fc259669c8a212a5cfdbd4feb0f5dccfff35e743
.. _995871e: https://github.com/pywavelet/pywavelet/commit/995871e367066164cb57d0bc34ab1d51fcfd9640
.. _5e59153: https://github.com/pywavelet/pywavelet/commit/5e59153d97227f4d108b27f4309ea26cb4031be7
.. _7976d20: https://github.com/pywavelet/pywavelet/commit/7976d20cf585ad62bb2b45d14e3be468f3825e35
.. _e6d88cd: https://github.com/pywavelet/pywavelet/commit/e6d88cd0b395492262bddf2741653354f94b9bf0
.. _cac0da9: https://github.com/pywavelet/pywavelet/commit/cac0da9575e5fc2591b92054e4b8bd4f9063eb20
.. _7093025: https://github.com/pywavelet/pywavelet/commit/709302534c0514c255a426ff70ea6601b6928729
.. _8ea7436: https://github.com/pywavelet/pywavelet/commit/8ea7436782cfd9fe468b9e9e58c722a9f525f530
.. _bd55639: https://github.com/pywavelet/pywavelet/commit/bd55639a5ae777b749822ccbe5737ecb3feba682
.. _3f763fb: https://github.com/pywavelet/pywavelet/commit/3f763fb98ba9adf2d608e09c094b4a32bd491d94
.. _e676e65: https://github.com/pywavelet/pywavelet/commit/e676e65d746be32d2b7a58349beece9512f4835e
.. _efc5b1f: https://github.com/pywavelet/pywavelet/commit/efc5b1f38eb0fd0f6094593684c50f8d6081078e
.. _0410893: https://github.com/pywavelet/pywavelet/commit/0410893fbac61b8ffb9bab896f1c63989a67823c
.. _02d984d: https://github.com/pywavelet/pywavelet/commit/02d984d17cc8b7dbcadea5b1cd05d8765e85f809
.. _980875b: https://github.com/pywavelet/pywavelet/commit/980875be202b5a21570d890c1c547175879f4108
.. _673f33c: https://github.com/pywavelet/pywavelet/commit/673f33cd5a11a84229944eea04a097c19a80cc1e
.. _c8c8f37: https://github.com/pywavelet/pywavelet/commit/c8c8f37dca50f1a9f3e05091d0c17123db00e373
.. _727c47d: https://github.com/pywavelet/pywavelet/commit/727c47dc18f656d36004ea2af6f2153b27f0188b
.. _7893845: https://github.com/pywavelet/pywavelet/commit/789384547dc81d3451640e0ee995ba8686267f29
.. _985e9eb: https://github.com/pywavelet/pywavelet/commit/985e9eba9880b4414cdb66d6cf95d060dde3f685
.. _1ee69b4: https://github.com/pywavelet/pywavelet/commit/1ee69b4b4d1470df2fa9d0971d4eea5075b5dc3f
.. _dae0fb0: https://github.com/pywavelet/pywavelet/commit/dae0fb06c4ae3361d19c85caa718505dbd7a8a20
.. _077e58e: https://github.com/pywavelet/pywavelet/commit/077e58ee8b7ab27d73991e5505d434149b2d58a6
.. _fd88319: https://github.com/pywavelet/pywavelet/commit/fd8831921dc3c66929e04eec117a52246bce77bd
.. _8552f77: https://github.com/pywavelet/pywavelet/commit/8552f77e7ae95e479e53295da4d20470f0e7bc4b
.. _3ee0be1: https://github.com/pywavelet/pywavelet/commit/3ee0be1c6da4894b677e8ca69c176e444274586f
.. _4411c74: https://github.com/pywavelet/pywavelet/commit/4411c74fea7f4c0e2f8e7cc6233e9b36550b74ae
.. _0a05d8d: https://github.com/pywavelet/pywavelet/commit/0a05d8d962e1d43446bdabd908a9dc7787aa056b
.. _68bf006: https://github.com/pywavelet/pywavelet/commit/68bf006905417445452133595168e24f75c03e0d
.. _53e1768: https://github.com/pywavelet/pywavelet/commit/53e1768aab02a457816f29ae6e54f6b35daeb9e9
.. _fd7628e: https://github.com/pywavelet/pywavelet/commit/fd7628e12eda2b171db9a6cdbb8727b653e33570
.. _ca545d4: https://github.com/pywavelet/pywavelet/commit/ca545d4e28ad2cb47e18c27b2494bf8a7eab7323
.. _77666f9: https://github.com/pywavelet/pywavelet/commit/77666f97a1b991d165211d715d2eed500cd688a2
.. _6253208: https://github.com/pywavelet/pywavelet/commit/62532080aafe0637d97da646cef461c3933aed78
.. _6ff8501: https://github.com/pywavelet/pywavelet/commit/6ff8501b2e49d7fa35dba59cb4f57a0e701a0bd2
.. _7b58b43: https://github.com/pywavelet/pywavelet/commit/7b58b43c99d9970e3fe0de59cc8dc35652059c47
.. _f73a3dc: https://github.com/pywavelet/pywavelet/commit/f73a3dcc4b9c8d92a302fc5287bb705caa301d39
.. _547fd32: https://github.com/pywavelet/pywavelet/commit/547fd326eaf3295e04359ece745b257472fcbd49
.. _5e5f2e1: https://github.com/pywavelet/pywavelet/commit/5e5f2e17ff3a54430899ea108572c7e351e3804c
.. _be9778f: https://github.com/pywavelet/pywavelet/commit/be9778f273f95f153dd03fbf872d1632aa630941
.. _6d704c0: https://github.com/pywavelet/pywavelet/commit/6d704c0ad99bdda26fbe3fca3fc6340e0842ba49
.. _a4083f4: https://github.com/pywavelet/pywavelet/commit/a4083f45ec33c577926cb3c394dd4ff4eb2ca945
.. _c77e2fe: https://github.com/pywavelet/pywavelet/commit/c77e2fe94bff8d44d111a1fbc24faf03b891a8e1
.. _a9a0610: https://github.com/pywavelet/pywavelet/commit/a9a061002dae29149826ce12930ba4fd20286548
.. _a811f24: https://github.com/pywavelet/pywavelet/commit/a811f243ad9856261fb9cba5e44dbef57aff7e76
.. _a0424ef: https://github.com/pywavelet/pywavelet/commit/a0424ef750bd5bbce232fa2f85da0ff4feb1def8
.. _a1cb77b: https://github.com/pywavelet/pywavelet/commit/a1cb77b6093ff0ebc5fb7bd342fd2e9f7ba7c39b
.. _2a1f889: https://github.com/pywavelet/pywavelet/commit/2a1f889cb89fca6e8ad77a236258389024a36620
.. _9d6b379: https://github.com/pywavelet/pywavelet/commit/9d6b379916137c59f526c931828db38a6629c3fb
.. _da3d090: https://github.com/pywavelet/pywavelet/commit/da3d0909ac48034725087ac84e8a236f69770095
.. _822d19b: https://github.com/pywavelet/pywavelet/commit/822d19b6021fc3d4b02fafeee3228d9105b083b8
.. _501fae1: https://github.com/pywavelet/pywavelet/commit/501fae1b67ee6186089964301c74c2bba7651771
.. _7c5f347: https://github.com/pywavelet/pywavelet/commit/7c5f347f73a83dc100081c4db7603de2fae67c67
.. _257f43d: https://github.com/pywavelet/pywavelet/commit/257f43dea6cd9324104a0b2dcc375388061b0228
.. _20a421d: https://github.com/pywavelet/pywavelet/commit/20a421de61172bb6a102699d4c8280be832674eb
.. _c3b9438: https://github.com/pywavelet/pywavelet/commit/c3b94387eb6fc2aea8195c1c8e74da25e86c530a
.. _206a5d7: https://github.com/pywavelet/pywavelet/commit/206a5d78b77c46cf98b3a75b6a09737524c9759b
.. _15949df: https://github.com/pywavelet/pywavelet/commit/15949dfb7d7956a57c6778d2998d204fb0a3827f
.. _a04a76e: https://github.com/pywavelet/pywavelet/commit/a04a76e49100cb6da0da50691b4c6e7e264d0502
.. _6572581: https://github.com/pywavelet/pywavelet/commit/657258152cb693cde2eef99106fb96c963671e22
.. _decfe7f: https://github.com/pywavelet/pywavelet/commit/decfe7f9ec29916b94dc7c455e604f423208bb85
.. _6209923: https://github.com/pywavelet/pywavelet/commit/620992301b140feee8e22f1566ada848cc35cc55
.. _49cb11c: https://github.com/pywavelet/pywavelet/commit/49cb11cd7ed95e78898066d0f150764dd59f53aa
.. _fda592d: https://github.com/pywavelet/pywavelet/commit/fda592d161ebd57565407adb1b8f3a5eb1ad7c09
.. _2f6810e: https://github.com/pywavelet/pywavelet/commit/2f6810e70fadd20f7e93c42a888fa2a635fceae9
.. _028349e: https://github.com/pywavelet/pywavelet/commit/028349e48fc2ffc2bef957f4f07fcc8d914a85af
.. _7cd06af: https://github.com/pywavelet/pywavelet/commit/7cd06af950ba7b8c3d06eb430da341cf8e0f3453
.. _7f4ece1: https://github.com/pywavelet/pywavelet/commit/7f4ece1b7622abf8b7dee525a75c3fbcc9a59adc
.. _ffea941: https://github.com/pywavelet/pywavelet/commit/ffea941d4ae29a64aff812c6c3c7aeafb2013b1a
.. _36a7279: https://github.com/pywavelet/pywavelet/commit/36a72790feb5540c538bfbef9ffd65d53cf00eba
.. _dae3912: https://github.com/pywavelet/pywavelet/commit/dae391293ad1349e41e9f3f8b4e5becb33fc19f5
.. _9c109d8: https://github.com/pywavelet/pywavelet/commit/9c109d83a8669336e6757d3de010c3ef9ebd9a45
.. _92c20fe: https://github.com/pywavelet/pywavelet/commit/92c20fed9facbc26211b952bdaf5705784c7ca31
.. _e15d5d3: https://github.com/pywavelet/pywavelet/commit/e15d5d3e0e0204a1679524ffe9073894b5e02c23
.. _c8e651c: https://github.com/pywavelet/pywavelet/commit/c8e651c6e682374f610446d3d9b9886759bcb6fd
.. _0d00f58: https://github.com/pywavelet/pywavelet/commit/0d00f584730648207f489a4fb99f672df670531f
.. _5927230: https://github.com/pywavelet/pywavelet/commit/59272302a8990c70009bd715b4b8f781aa24a16e
.. _141cfac: https://github.com/pywavelet/pywavelet/commit/141cfac5ee5e1186ef0e9b8ed4dde7e839e1609c
.. _b709b9e: https://github.com/pywavelet/pywavelet/commit/b709b9ed269b813d28dd84329868dbcd710a682f
.. _67948ed: https://github.com/pywavelet/pywavelet/commit/67948ed014227a5eac9114e36ae49312e20d363a
.. _d50e6e8: https://github.com/pywavelet/pywavelet/commit/d50e6e861058362e3143f81072b164930c323520
.. _7141a73: https://github.com/pywavelet/pywavelet/commit/7141a7354fde30743626d0af2cec76b3bf56dacd
.. _01c6050: https://github.com/pywavelet/pywavelet/commit/01c6050fc792b2d0df4515062eea328057827b18
.. _20f5d30: https://github.com/pywavelet/pywavelet/commit/20f5d301e024a1693682428cb5c6c8cd96f561e5
.. _0cdf9c1: https://github.com/pywavelet/pywavelet/commit/0cdf9c13fbafec597261d808b9ce5ec0e8885d20
.. _15e24f0: https://github.com/pywavelet/pywavelet/commit/15e24f00a67a48f21daa7f0540bff533f1cebe8e
.. _d65d993: https://github.com/pywavelet/pywavelet/commit/d65d993b7c71750e1fad6b742e5e06ffcb191cb7
.. _9024797: https://github.com/pywavelet/pywavelet/commit/9024797b032ffc0490661d0e88a9c679d9ddd9ff
.. _a04112e: https://github.com/pywavelet/pywavelet/commit/a04112ed4c59cfdcfc2dc793c19d202e96d32df2
.. _4024ae6: https://github.com/pywavelet/pywavelet/commit/4024ae66eee5db795054de3e550a695e5c5cb6b2
.. _78a1f73: https://github.com/pywavelet/pywavelet/commit/78a1f739a41f0d4dbaae03ff53d77db45c14a13f
.. _bc50c43: https://github.com/pywavelet/pywavelet/commit/bc50c4352a5177b7ba2914fdac7f97ce25baa37b
.. _68b3eec: https://github.com/pywavelet/pywavelet/commit/68b3eecb2a110aa4191e5dd523c31c1560b835e4
.. _0cec53c: https://github.com/pywavelet/pywavelet/commit/0cec53c2e81b1f5d3701d09bc97bdf93a1af7eaf
.. _29665f5: https://github.com/pywavelet/pywavelet/commit/29665f58f99442f25ae3d652913c0dd6794ca7ab
.. _5dd5e0f: https://github.com/pywavelet/pywavelet/commit/5dd5e0f933b56606c64be78a2f806a2224506eef
.. _416c810: https://github.com/pywavelet/pywavelet/commit/416c8100142c60558540b65cdbb9b8c452be81f7
.. _d07ae7e: https://github.com/pywavelet/pywavelet/commit/d07ae7e8213af68affe6deb24602df1751917bf5
.. _113251b: https://github.com/pywavelet/pywavelet/commit/113251b4746c4f4718f0dfb078031a159509445c
.. _0c1820d: https://github.com/pywavelet/pywavelet/commit/0c1820dd06597635cda00ef1c210f8c0fcda2d5b
.. _e2b3767: https://github.com/pywavelet/pywavelet/commit/e2b37678ac6772a9909ca714127ce75338a926ee
.. _b1947f0: https://github.com/pywavelet/pywavelet/commit/b1947f05b9daedc375d8748b95b123ac3b5cb857
.. _1596bde: https://github.com/pywavelet/pywavelet/commit/1596bdeacec68fa932fcf2bb648fa30da1e6984d
.. _0ca80ee: https://github.com/pywavelet/pywavelet/commit/0ca80ee2ee388f52b2512a05d489544a634b4fb4
.. _35d2ce2: https://github.com/pywavelet/pywavelet/commit/35d2ce2d8bb6099efced927a3dfbabff8cc7732c
.. _ad43d13: https://github.com/pywavelet/pywavelet/commit/ad43d13f110e11d67fe79ae1fae168b85a350554
.. _b2db4b3: https://github.com/pywavelet/pywavelet/commit/b2db4b3a6654b641e4b3588d8db579378c52df05
.. _3fbbaf4: https://github.com/pywavelet/pywavelet/commit/3fbbaf4332fa62c01b04671914f4ed0b25a3096e
.. _d13f219: https://github.com/pywavelet/pywavelet/commit/d13f219a9f35e0566e6c23a4d048500fe23fa91c
.. _7add237: https://github.com/pywavelet/pywavelet/commit/7add23717940d5c0ff40f7be33f1d979927ef37b
.. _f9fc53b: https://github.com/pywavelet/pywavelet/commit/f9fc53b1347452ff198361103984bc97fa03be05
.. _5a5dff2: https://github.com/pywavelet/pywavelet/commit/5a5dff2453c53c7b20f3233628f3b9f6b510a918
.. _df6016e: https://github.com/pywavelet/pywavelet/commit/df6016e8f8ffbfa623a442d1a0450225394c4aaf
.. _13d7dce: https://github.com/pywavelet/pywavelet/commit/13d7dce3c6b62f4e18671e9aada92f24321fd8e1
.. _31770ec: https://github.com/pywavelet/pywavelet/commit/31770ecd9c59aa1ae8e21402be40bb0a494912aa
.. _e50827b: https://github.com/pywavelet/pywavelet/commit/e50827bd7bfd8d2ebbfaa1fd9b9e76dca563e20c
.. _12776a4: https://github.com/pywavelet/pywavelet/commit/12776a4b6c08fbef57a09598b7f4f29ea2afa018
.. _379bad7: https://github.com/pywavelet/pywavelet/commit/379bad7fa55b731051ab08f4ae6314dc426979b8
.. _73e9d42: https://github.com/pywavelet/pywavelet/commit/73e9d4233a9fb02cc751d61b038c60615b495645
.. _1f542bc: https://github.com/pywavelet/pywavelet/commit/1f542bcdb4d0a9a57b8386d25275293544411c18
.. _631ab0c: https://github.com/pywavelet/pywavelet/commit/631ab0cc4c63085e1dab5e609072d9c5baf94206
.. _7b88b52: https://github.com/pywavelet/pywavelet/commit/7b88b522c97ca2bbb8cad9bf24879d07e34799e1
.. _750b709: https://github.com/pywavelet/pywavelet/commit/750b7098d17916fdaa760ca14beba0beac19943e
.. _64c12c9: https://github.com/pywavelet/pywavelet/commit/64c12c9244813431cc0be6c7f5db4ee88925b17c
.. _70e6362: https://github.com/pywavelet/pywavelet/commit/70e636292802a607d564d95e090661445144bdbe
.. _e664c48: https://github.com/pywavelet/pywavelet/commit/e664c48031266084c7016cb8ee2facf1c234c6b4
.. _39ce268: https://github.com/pywavelet/pywavelet/commit/39ce2681afacc2c0191d9579850beae5a26031a3
.. _32a3998: https://github.com/pywavelet/pywavelet/commit/32a39980101438671a85d85bda518320718128e3
.. _fe01f91: https://github.com/pywavelet/pywavelet/commit/fe01f911bd79d33322edb8e24920bd504097072d
.. _3f5b34c: https://github.com/pywavelet/pywavelet/commit/3f5b34ca8e4d6c0f7b9f97b2e0e7c54e71de4f13
.. _4c864e2: https://github.com/pywavelet/pywavelet/commit/4c864e244f92a2fa12dcd82cadcd403f3e9055c5
.. _39f7526: https://github.com/pywavelet/pywavelet/commit/39f752617ff0dddd40aba826f4ed9983b464d371
.. _bbe764f: https://github.com/pywavelet/pywavelet/commit/bbe764fd3e1d60cf809449bf52d77a351d6ace4d
.. _56664c3: https://github.com/pywavelet/pywavelet/commit/56664c3486cad789159e718918a8019b46de9e90
.. _c1f4f92: https://github.com/pywavelet/pywavelet/commit/c1f4f929a14149430a63580181d62992b5b45be0
.. _c62bcde: https://github.com/pywavelet/pywavelet/commit/c62bcde597f328bc43185089f0286460ea4f9046
.. _1ca831c: https://github.com/pywavelet/pywavelet/commit/1ca831c4a50baf81ff44b18593184c26e93557a4
.. _30d8444: https://github.com/pywavelet/pywavelet/commit/30d8444cc4967f8cfef9bdab1c008ed933456fe1
.. _f502346: https://github.com/pywavelet/pywavelet/commit/f5023462b7df88f8ace09d8e50b787994615efcc
.. _f204ad1: https://github.com/pywavelet/pywavelet/commit/f204ad19b6fae6375a9148afd413faf2ec17cc95
.. _5c2b2f4: https://github.com/pywavelet/pywavelet/commit/5c2b2f4538f564d206a77607e8929a34a349c44b
.. _62d2fd6: https://github.com/pywavelet/pywavelet/commit/62d2fd6af86e43ba4e7997210dadef9684ca6830
.. _d04c1e4: https://github.com/pywavelet/pywavelet/commit/d04c1e40f2176a6535c6bcedbfd23a6f5d7a315e
.. _26ba587: https://github.com/pywavelet/pywavelet/commit/26ba5874d5f77cdaab5d171184282ecab5810f82
.. _0321216: https://github.com/pywavelet/pywavelet/commit/032121643e522a1423223583ffce5b2c3b1daea3
.. _fa5dab0: https://github.com/pywavelet/pywavelet/commit/fa5dab0eebf38cfd708cfd2feda98e7b5eaccb0c
.. _a42daaf: https://github.com/pywavelet/pywavelet/commit/a42daaf79edf34fa2b99a62d0180f9070902f01e
.. _9213db2: https://github.com/pywavelet/pywavelet/commit/9213db20fc2e7de23cdaebc88b1f407325ee0c2b
.. _45c6aa3: https://github.com/pywavelet/pywavelet/commit/45c6aa34094f042d77a10c214d264f0707556dec
.. _2d64fbe: https://github.com/pywavelet/pywavelet/commit/2d64fbe46b4838d57068e35c4fead80f87ca48bf
.. _e495a59: https://github.com/pywavelet/pywavelet/commit/e495a597c27a19335d69e453ce2e7a4bbe76b610
.. _f37b03e: https://github.com/pywavelet/pywavelet/commit/f37b03eca51828e260e675811f2936a6eb2e147b
.. _189c510: https://github.com/pywavelet/pywavelet/commit/189c51066520151df2910ba2acf2a19ab0cf2dec
.. _79f4e0e: https://github.com/pywavelet/pywavelet/commit/79f4e0eb59d619a703eece52b94cdcdf7a6178b3
.. _37869e6: https://github.com/pywavelet/pywavelet/commit/37869e659aeb3cc73eee3ecf60732bf36b08f142
.. _97a0402: https://github.com/pywavelet/pywavelet/commit/97a0402ef9c1b68281fe4984f8ce559d5df71546
.. _bfd3a13: https://github.com/pywavelet/pywavelet/commit/bfd3a13c34e3409b09dabda176aca7902fc05b7f
.. _39119b4: https://github.com/pywavelet/pywavelet/commit/39119b4e25c8e018b92aa37589a56b3d4f7f6caf
.. _02fcc81: https://github.com/pywavelet/pywavelet/commit/02fcc81180341ecfb2ec36401966f4bf7e56dcb0
