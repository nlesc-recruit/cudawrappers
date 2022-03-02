**Description**

<!-- Description of PR -->

**Related issues**:

- ...

**Instructions to review the pull request**

- Check that CHANGELOG.md has been updated if necessary

<!--
Clone and verify
```
cd $(mktemp -d --tmpdir cudawrappers-XXXXXX)
git clone https://github.com/nlesc-recruit/CUDA-wrappers .
git checkout <this-branch>
cmake -S . -B build
make -C build format # Nothing should change
make -C build lint # linting is broken until we fix the issues (see #92)
```
-->

<!--
Review online.
-->
